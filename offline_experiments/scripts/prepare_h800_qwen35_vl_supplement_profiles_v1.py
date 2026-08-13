#!/usr/bin/env python3
"""Build processor-bound profile aliases for the Qwen3.5/VL supplement.

The expensive image-grid computation is performed once per processor family.
An alias is permitted only when the tokenizer and processor configuration
fingerprints are byte-identical to the representative checkpoint.  This is a
pre-GPU materialization step and never authorizes or launches training.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from common import ARTIFACT_DIR, sha256_file, sha256_json, write_json
import prepare_h800_vl_calibration_profiles_v1 as base


CAMPAIGN_ID = "h800_qwen35_vl_supplement_20260809_v1"
OUTPUT_DIR = ARTIFACT_DIR / "h800_qwen35_vl_supplement_profiles_v1"
OUTPUT_MANIFEST = (
    ARTIFACT_DIR / "h800_qwen35_vl_supplement_processor_profiles_manifest_v1.json"
)

FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "id": "qwen2p5_vl",
        "representative_id": "qwen2p5_vl_7b",
        "representative_path": "/wanqing-models/Qwen2.5-VL-7B-Instruct",
        "template": "qwen2_vl",
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_min_pixels": 56 * 56,
        "members": (
            ("qwen2p5_vl_3b", "/wanqing-models/Qwen2.5-VL-3B-Instruct"),
            ("qwen2p5_vl_7b", "/wanqing-models/Qwen2.5-VL-7B-Instruct"),
        ),
    },
    {
        "id": "qwen3_vl",
        "representative_id": "qwen3_vl_8b",
        "representative_path": "/wanqing-models/Qwen3-VL-8B-Instruct",
        "template": "qwen3_vl",
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_min_pixels": 64 * 64,
        "members": (
            ("qwen3_vl_2b", "/wanqing-models/Qwen3-VL-2B-Instruct"),
            ("qwen3_vl_4b", "/wanqing-models/Qwen3-VL-4B-Instruct"),
            ("qwen3_vl_8b", "/wanqing-models/Qwen3-VL-8B-Instruct"),
            ("qwen3_vl_32b", "/wanqing-models/Qwen3-VL-32B-Instruct"),
            ("qwen3_vl_30b_a3b", "/wanqing-models/Qwen3-VL-30B-A3B-Instruct"),
        ),
    },
    {
        "id": "qwen3_5",
        "representative_id": "qwen3p5_4b",
        "representative_path": "/wanqing-models/Qwen3.5-4B",
        "template": "qwen3_5_nothink",
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "image_min_pixels": 64 * 64,
        "members": (
            ("qwen3p5_0p8b", "/wanqing-models/Qwen3.5-0.8B"),
            ("qwen3p5_4b", "/wanqing-models/Qwen3.5-4B"),
            ("qwen3p5_9b", "/wanqing-models/Qwen3.5-9B"),
            ("qwen3p5_27b", "/wanqing-models/Qwen3.5-27B"),
        ),
    },
)

FINGERPRINT_FILES = (
    "tokenizer.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _fingerprint(path: Path) -> dict[str, str]:
    files = {
        name: sha256_file(path / name)
        for name in FINGERPRINT_FILES
        if (path / name).is_file()
    }
    if not files:
        raise FileNotFoundError(f"no tokenizer/processor files found in {path}")
    return files


def _materialize_representative(family: dict[str, Any]) -> dict[str, Any]:
    base.SCHEMA = "sft_h800_qwen35_vl_supplement_representative_profiles/v1"
    base.MODELS = (
        {
            "id": family["representative_id"],
            "path": family["representative_path"],
            "template": family["template"],
            "patch_size": family["patch_size"],
            "merge_size": family["merge_size"],
            "temporal_patch_size": family["temporal_patch_size"],
            "image_min_pixels": family["image_min_pixels"],
        },
    )
    base.OUTPUT_DIR = OUTPUT_DIR / "representatives" / family["id"]
    base.OUTPUT_MANIFEST = OUTPUT_DIR / "representatives" / f"{family['id']}.json"
    if base.OUTPUT_MANIFEST.is_file():
        cached = json.loads(base.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
        cached_rows = cached.get("profiles") or []
        if (
            cached.get("all_actual_processor_checks_passed") is True
            and len(cached_rows) == 2
            and all(
                Path(row["path"]).is_file()
                and row["sha256"] == sha256_file(Path(row["path"]))
                for row in cached_rows
            )
        ):
            return cached
    return base.prepare()


def _alias_profile(
    *, source_path: Path, model_id: str, family: dict[str, Any], tier: str
) -> dict[str, Any]:
    profile = copy.deepcopy(json.loads(source_path.read_text(encoding="utf-8")))
    profile.pop("report_sha256", None)
    profile["model"] = {"model_id": model_id}
    profile["profile_alias"] = {
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "representative_model_id": family["representative_id"],
        "equivalence_policy": (
            "canonical tokenizer.json and image/video processor files are byte-identical; "
            "the fixed LlamaFactory template therefore has shared tokenization and image-grid geometry"
        ),
    }
    profile["campaign_id"] = CAMPAIGN_ID
    profile["tier"] = tier
    profile["gpu_training_started"] = False
    profile["report_sha256"] = sha256_json(profile)
    destination = OUTPUT_DIR / f"{model_id}.{tier}.json"
    write_json(destination, profile)
    return {
        "model_id": model_id,
        "family": family["id"],
        "tier": tier,
        "image_min_pixels": family["image_min_pixels"],
        "image_max_pixels": 448 * 448 if tier == "low" else 768 * 768,
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "report_sha256": profile["report_sha256"],
        "summary": profile["summary"],
        "actual_processor_validation": profile["actual_processor_validation"],
    }


def prepare() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    representative_manifests: list[dict[str, Any]] = []
    for family in FAMILIES:
        representative_path = Path(family["representative_path"])
        representative_fingerprint = _fingerprint(representative_path)
        member_fingerprints = {}
        for model_id, path_text in family["members"]:
            path = Path(path_text)
            fingerprint = _fingerprint(path)
            if fingerprint != representative_fingerprint:
                raise ValueError(
                    f"{model_id} is not processor/tokenizer-equivalent to "
                    f"{family['representative_id']}"
                )
            member_fingerprints[model_id] = fingerprint
        generated = _materialize_representative(family)
        representative_manifests.append(
            {
                "family": family["id"],
                "path": str(base.OUTPUT_MANIFEST.resolve()),
                "sha256": sha256_file(base.OUTPUT_MANIFEST),
                "report_sha256": generated["report_sha256"],
            }
        )
        for tier in ("low", "high"):
            source_path = base.OUTPUT_DIR / f"{family['representative_id']}.{tier}.json"
            for model_id, _ in family["members"]:
                rows.append(
                    _alias_profile(
                        source_path=source_path,
                        model_id=model_id,
                        family=family,
                        tier=tier,
                    )
                )
        equivalence.append(
            {
                "family": family["id"],
                "representative_model_id": family["representative_id"],
                "representative_fingerprint": representative_fingerprint,
                "representative_fingerprint_sha256": sha256_json(
                    representative_fingerprint
                ),
                "member_model_ids": [model_id for model_id, _ in family["members"]],
                "all_member_fingerprints_exact": True,
                "member_fingerprints": member_fingerprints,
            }
        )
    report: dict[str, Any] = {
        "schema": "sft_h800_qwen35_vl_supplement_processor_profiles/v1",
        "campaign_id": CAMPAIGN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "profiles": rows,
        "processor_family_equivalence": equivalence,
        "representative_manifests": representative_manifests,
        "all_actual_processor_checks_passed": all(
            row["actual_processor_validation"]["all_exact"] is True for row in rows
        ),
        "all_alias_equivalence_checks_passed": True,
        "usage": "calibration_fit_only_never_prospective_acceptance",
    }
    report["report_sha256"] = sha256_json(report)
    write_json(OUTPUT_MANIFEST, report)
    return report


def main() -> None:
    report = prepare()
    print(
        json.dumps(
            {
                "manifest": str(OUTPUT_MANIFEST),
                "profiles": len(report["profiles"]),
                "families": len(report["processor_family_equivalence"]),
                "all_actual_processor_checks_passed": report[
                    "all_actual_processor_checks_passed"
                ],
                "all_alias_equivalence_checks_passed": report[
                    "all_alias_equivalence_checks_passed"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
