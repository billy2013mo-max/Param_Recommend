#!/usr/bin/env python3
"""Stable H800 V3-memory plus V5-throughput resource predictor."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common import write_json
from h800_physical_v4b_predictor import (
    DEFAULT_MEMORY_ANCHOR_REGISTRY,
    _load_requests,
)
from h800_physical_v4b_predictor import (
    DEFAULT_MEMORY_ARTIFACT as DEFAULT_LEGACY_MEMORY_ARTIFACT,
)
from h800_physical_v4b_predictor import (
    DEFAULT_THROUGHPUT_ARTIFACT as DEFAULT_LEGACY_THROUGHPUT_ARTIFACT,
)
from h800_physical_v4b_predictor import (
    validate_prediction_report as validate_legacy_prediction_report,
)
from h800_unified_v3_throughput_v5_predictor import (
    DEFAULT_THROUGHPUT_ARTIFACT,
    H800UnifiedV3ThroughputV5Predictor,
)
from h800_unified_v3_throughput_v5_predictor import (
    SCHEMA as ACTIVE_PREDICTION_SCHEMA,
)
from h800_unified_v3_throughput_v5_predictor import (
    validate_prediction_report as validate_active_prediction_report,
)
from h800_unified_v3_v4b_predictor import (
    DEFAULT_MEMORY_ARTIFACT as DEFAULT_V3_MEMORY_ARTIFACT,
)
from h800_unified_v3_v4b_predictor import (
    DEFAULT_MEMORY_FEATURE_INVENTORY,
    H800LegacyRollbackPredictor,
)
from h800_unified_v3_v4b_predictor import (
    SCHEMA as HISTORICAL_V3_V4B_PREDICTION_SCHEMA,
)
from h800_unified_v3_v4b_predictor import (
    validate_prediction_report as validate_historical_v3_v4b_report,
)
from throughput_predictor import DEFAULT_MODEL_INVENTORY

DEFAULT_MEMORY_GATE = "unified_v3"
MEMORY_GATES = ("unified_v3", "legacy_physical_v1")


def validate_prediction_report(report: Mapping[str, Any]) -> None:
    """Dispatch validation by the report's versioned schema."""

    if report.get("schema") == ACTIVE_PREDICTION_SCHEMA:
        validate_active_prediction_report(report)
    elif report.get("schema") == HISTORICAL_V3_V4B_PREDICTION_SCHEMA:
        validate_historical_v3_v4b_report(report)
    else:
        validate_legacy_prediction_report(report)


class H800ResourcePredictor:
    """Canonical predictor: memory V3 plus throughput V5 by default."""

    def __init__(
        self,
        *,
        memory_gate: str = DEFAULT_MEMORY_GATE,
        memory_artifact: Path | None = None,
        legacy_memory_artifact: Path = DEFAULT_LEGACY_MEMORY_ARTIFACT,
        memory_feature_inventory: Path = DEFAULT_MEMORY_FEATURE_INVENTORY,
        throughput_artifact: Path = DEFAULT_THROUGHPUT_ARTIFACT,
        legacy_throughput_artifact: Path = DEFAULT_LEGACY_THROUGHPUT_ARTIFACT,
        memory_anchor_registry: Path = DEFAULT_MEMORY_ANCHOR_REGISTRY,
        model_inventory: Path = DEFAULT_MODEL_INVENTORY,
        strict_model_inventory_binding: bool = True,
        additional_dataset_profile_dir: Path | None = None,
    ) -> None:
        gate = str(memory_gate)
        if gate not in MEMORY_GATES:
            raise ValueError(f"memory_gate must be one of {MEMORY_GATES}")
        self.memory_gate = gate
        if gate == "unified_v3":
            self.predictor = H800UnifiedV3ThroughputV5Predictor(
                memory_artifact=(memory_artifact or DEFAULT_V3_MEMORY_ARTIFACT),
                memory_feature_inventory=memory_feature_inventory,
                throughput_artifact=throughput_artifact,
                model_inventory=model_inventory,
                strict_model_inventory_binding=strict_model_inventory_binding,
                additional_dataset_profile_dir=additional_dataset_profile_dir,
            )
        else:
            self.predictor = H800LegacyRollbackPredictor(
                memory_artifact=(memory_artifact or legacy_memory_artifact),
                throughput_artifact=legacy_throughput_artifact,
                memory_anchor_registry=memory_anchor_registry,
                model_inventory=model_inventory,
                strict_model_inventory_binding=strict_model_inventory_binding,
                additional_dataset_profile_dir=additional_dataset_profile_dir,
            )

    def predict(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self.predictor.predict(requests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--memory-gate",
        choices=MEMORY_GATES,
        default=DEFAULT_MEMORY_GATE,
        help=(
            "Default is memory V3 plus throughput V5; legacy_physical_v1 "
            "rolls back the whole legacy memory-plus-v4b pipeline."
        ),
    )
    parser.add_argument("--memory-artifact", type=Path, default=None)
    parser.add_argument(
        "--legacy-memory-artifact",
        type=Path,
        default=DEFAULT_LEGACY_MEMORY_ARTIFACT,
    )
    parser.add_argument(
        "--memory-feature-inventory",
        type=Path,
        default=DEFAULT_MEMORY_FEATURE_INVENTORY,
    )
    parser.add_argument(
        "--throughput-artifact",
        type=Path,
        default=DEFAULT_THROUGHPUT_ARTIFACT,
    )
    parser.add_argument(
        "--legacy-throughput-artifact",
        type=Path,
        default=DEFAULT_LEGACY_THROUGHPUT_ARTIFACT,
    )
    parser.add_argument(
        "--memory-anchor-registry",
        type=Path,
        default=DEFAULT_MEMORY_ANCHOR_REGISTRY,
    )
    parser.add_argument("--additional-dataset-profile-dir", type=Path, default=None)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_MODEL_INVENTORY)
    parser.add_argument("--allow-unfrozen-model-inventory", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    predictor = H800ResourcePredictor(
        memory_gate=args.memory_gate,
        memory_artifact=args.memory_artifact,
        legacy_memory_artifact=args.legacy_memory_artifact,
        memory_feature_inventory=args.memory_feature_inventory,
        throughput_artifact=args.throughput_artifact,
        legacy_throughput_artifact=args.legacy_throughput_artifact,
        memory_anchor_registry=args.memory_anchor_registry,
        model_inventory=args.model_inventory,
        strict_model_inventory_binding=not args.allow_unfrozen_model_inventory,
        additional_dataset_profile_dir=args.additional_dataset_profile_dir,
    )
    report = predictor.predict(_load_requests(args.input))
    validate_prediction_report(report)
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
