#!/usr/bin/env python3
"""Freeze 20 new source-disjoint datasets for H800 LoRA safety stage two.

This CPU-only step selects already downloaded and audited datasets from
``infra-ai-infra-storage/datasets/``.  It enforces zero exact normalized-row
overlap among the 20 selections and against all earlier 19 current sources and
the five connected historical datasets.  No GPU process is launched.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers.utils import import_utils as transformers_import_utils

# This CPU-only preparation process never uses TorchAO quantization.  The shared
# launcher currently contains a TorchAO build that targets a newer PyTorch than
# the launcher's pinned 2.8 runtime, so prevent Transformers from importing that
# optional backend while loading LLaMA Factory's text template implementation.
transformers_import_utils._torchao_available = False

from common import ARTIFACT_DIR, DATA_DIR, ROOT, percentile, sha256_file, sha256_json, write_json
from prepare_h800_lora_source_disjoint_data_v1 import (
    MAX_SELECTED_ROWS,
    REMOTE_BUCKET,
    REMOTE_PREFIX,
    TEMPLATE,
    DatasetSpec,
    TrainingEncoder,
    _atomic_jsonl,
    _fingerprint,
    _iter_objects,
    _latest_remote_revisions,
    _normalize,
    _ratio_bin,
    _round_up,
    _tokenizer_binding,
)


SCHEMA = "sft_h800_lora_safety_stage2_data_bundle/v1"
SELECTION_COUNT = 20
DEFAULT_AUDIT = ARTIFACT_DIR / "h800_lora_remote_candidate_audit_v1.json"
DEFAULT_REMOTE_LISTING = Path("/tmp/ai_infra_storage_datasets.tsv")
DEFAULT_STAGE1_BUNDLE = ARTIFACT_DIR / "h800_lora_source_disjoint_bundle_v1.json"
DEFAULT_STAGE1_SPLIT = (
    ARTIFACT_DIR / "h800_lora_source_disjoint_v1" / "split_manifest.json"
)
DEFAULT_OLD_DESIGN = (
    ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json"
)
DEFAULT_HISTORICAL_ANALYSIS = ARTIFACT_DIR / "dataset_analysis.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_lora_safety_stage2_bundle_v1.json"
DATA_OUTPUT_DIR = DATA_DIR / "h800_lora_safety_stage2_v1"
PROFILE_DIR = ARTIFACT_DIR / "h800_lora_safety_stage2_v1" / "profiles"
PROCESSOR_CONTRACT = (
    ARTIFACT_DIR / "h800_lora_safety_stage2_v1" / "processor_contract.json"
)
SPLIT_MANIFEST = (
    ARTIFACT_DIR / "h800_lora_safety_stage2_v1" / "split_manifest.json"
)


SPECS = (
    DatasetSpec(
        "dataset-ijyfni-1783330528",
        "lora_s2_src01_live_pk_script",
        "qwen3_14b",
        Path("/wanqing-models/Qwen3-14B"),
        2048,
        "le_0p40",
        "直播PK催票话术生成",
    ),
    DatasetSpec(
        "dataset-m5j09g-1784713977",
        "lora_s2_src02_comment_intent",
        "qwen3_14b",
        Path("/wanqing-models/Qwen3-14B"),
        2048,
        "le_0p40",
        "平台评论意图分类",
    ),
    DatasetSpec(
        "dataset-ac19v1-1782985558",
        "lora_s2_src03_quality_comment",
        "qwen3_14b",
        Path("/wanqing-models/Qwen3-14B"),
        2048,
        "le_0p40",
        "优质评论识别",
    ),
    DatasetSpec(
        "dataset-63v7nn-1785673698",
        "lora_s2_src04_live_punishment",
        "qwen3_14b",
        Path("/wanqing-models/Qwen3-14B"),
        2048,
        "le_0p40",
        "直播PK惩罚偏好识别",
    ),
    DatasetSpec(
        "dataset-yysg3v-1780379911",
        "lora_s2_src05_beauty_slots",
        "qwen3_14b",
        Path("/wanqing-models/Qwen3-14B"),
        4096,
        "le_0p40",
        "直播美化需求槽位抽取",
    ),
    DatasetSpec(
        "dataset-ojs5cu-1785230598",
        "lora_s2_src06_voice_intent",
        "qwen3_14b",
        Path("/wanqing-models/Qwen3-14B"),
        1536,
        "gt_0p40_le_0p70",
        "语音助手意图二分类",
    ),
    DatasetSpec(
        "dataset-odjfiy-1783140891",
        "lora_s2_src07_product_facts",
        "qwen3_14b",
        Path("/wanqing-models/Qwen3-14B"),
        2560,
        "gt_0p40_le_0p70",
        "生活服务商品事实抽取",
    ),
    DatasetSpec(
        "dataset-spt19j-1784396541",
        "lora_s2_src08_flood_topic",
        "qwen3_8b",
        Path("/wanqing-models/Qwen3-8B"),
        4096,
        "gt_0p40_le_0p70",
        "汛期热点专题分类",
    ),
    DatasetSpec(
        "dataset-nf1f4n-1784029736",
        "lora_s2_src09_superstition_audit",
        "qwen3_8b",
        Path("/wanqing-models/Qwen3-8B"),
        4096,
        "gt_0p40_le_0p70",
        "封建迷信内容审核",
    ),
    DatasetSpec(
        "dataset-ylrzb1-1784856535",
        "lora_s2_src10_flood_route",
        "qwen3_8b",
        Path("/wanqing-models/Qwen3-8B"),
        4096,
        "gt_0p40_le_0p70",
        "汛期热点分类路由",
    ),
    DatasetSpec(
        "dataset-3vi9rj-1784861980",
        "lora_s2_src11_flood_route_fallback",
        "qwen3_8b",
        Path("/wanqing-models/Qwen3-8B"),
        3072,
        "gt_0p70_le_0p90",
        "汛期热点分类与证据不足回退",
    ),
    DatasetSpec(
        "dataset-kuxhnp-1784796471",
        "lora_s2_src12_flood_multilabel",
        "qwen3_8b",
        Path("/wanqing-models/Qwen3-8B"),
        4608,
        "gt_0p70_le_0p90",
        "汛期热词多标签分类",
    ),
    DatasetSpec(
        "dataset-w3nrzb-1785738899",
        "lora_s2_src13_text_safety",
        "qwen3_8b",
        Path("/wanqing-models/Qwen3-8B"),
        5120,
        "gt_0p70_le_0p90",
        "文本安全合规审核",
    ),
    DatasetSpec(
        "dataset-9jtmeo-1785231016",
        "lora_s2_src14_ad_metrics",
        "qwen3_4b",
        Path("/wanqing-models/Qwen3-4B"),
        6144,
        "gt_0p70_le_0p90",
        "广告投放数据分析",
    ),
    DatasetSpec(
        "dataset-fjbx43-1780048284",
        "lora_s2_src15_service_quality",
        "qwen3_4b",
        Path("/wanqing-models/Qwen3-4B"),
        6144,
        "gt_0p70_le_0p90",
        "客服对话质量评分",
    ),
    DatasetSpec(
        "dataset-dkdq2n-1785422354",
        "lora_s2_src16_product_match",
        "qwen3_4b",
        Path("/wanqing-models/Qwen3-4B"),
        8192,
        "gt_0p95",
        "广告视频商品匹配",
    ),
    DatasetSpec(
        "dataset-u5curm-1781773394",
        "lora_s2_src17_content_risk",
        "qwen3_4b",
        Path("/wanqing-models/Qwen3-4B"),
        8192,
        "gt_0p95",
        "多模态转写内容风险分类",
    ),
    DatasetSpec(
        "dataset-rxm7if-1784712016",
        "lora_s2_src18_marketing_antifraud",
        "qwen3_4b",
        Path("/wanqing-models/Qwen3-4B"),
        8192,
        "gt_0p95",
        "电商营销行为反作弊",
    ),
    DatasetSpec(
        "dataset-ipceq3-1784029124",
        "lora_s2_src19_video_ad_audit",
        "qwen3_4b",
        Path("/wanqing-models/Qwen3-4B"),
        8192,
        "gt_0p95",
        "视频广告违规分类",
    ),
    DatasetSpec(
        "dataset-uqxipt-1785725891",
        "lora_s2_src20_hotword_rewrite",
        "qwen3_4b",
        Path("/wanqing-models/Qwen3-4B"),
        8192,
        "gt_0p95",
        "热点文案改写",
    ),
)


def _universal_fingerprint(value: Any, *, path: Path) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{path} contains a non-object row")
    if all(isinstance(value.get(field), str) for field in ("system", "prompt", "response")):
        return _fingerprint(_normalize(value, path=path))
    messages = value.get("messages")
    if isinstance(messages, list):
        systems: list[str] = []
        prompts: list[str] = []
        responses: list[str] = []
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                continue
            role = str(message.get("role") or "").lower()
            if role == "system":
                systems.append(message["content"])
            elif role in {"user", "human"}:
                prompts.append(message["content"])
            elif role in {"assistant", "gpt"}:
                responses.append(message["content"])
        if prompts and responses:
            normalized = {
                "system": "\n".join(systems),
                "prompt": "\n".join(prompts),
                "response": "\n".join(responses),
            }
            return _fingerprint(normalized)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprints(path: Path) -> set[str]:
    return {
        _universal_fingerprint(value, path=path)
        for value in _iter_objects(path)
    }


def _prior_sources(
    *,
    stage1_split_path: Path,
    old_design_path: Path,
    historical_analysis_path: Path,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    stage1 = json.loads(stage1_split_path.read_text(encoding="utf-8"))
    for row in stage1["selection_rows"]:
        result[f"stage1:{row['source_dataset_id']}"] = _fingerprints(
            Path(row["source_path"])
        )
    old = json.loads(old_design_path.read_text(encoding="utf-8"))
    for scenario in old["scenarios"]:
        result[f"current_old:{scenario['dataset_id']}"] = _fingerprints(
            Path(scenario["data_path"])
        )
    historical = json.loads(historical_analysis_path.read_text(encoding="utf-8"))
    for dataset_id in (
        "short_512",
        "multiturn_2048",
        "multiturn_4096",
        "longtail_8192",
        "longcontext_32768",
    ):
        result[f"historical:{dataset_id}"] = _fingerprints(
            Path(historical["datasets"][dataset_id]["file"])
        )
    return result


def prepare(
    *,
    audit_path: Path,
    remote_listing: Path,
    stage1_bundle_path: Path,
    stage1_split_path: Path,
    old_design_path: Path,
    historical_analysis_path: Path,
) -> dict[str, Any]:
    if len(SPECS) != SELECTION_COUNT:
        raise ValueError(f"expected {SELECTION_COUNT} specs, got {len(SPECS)}")
    if len({spec.dataset_id for spec in SPECS}) != SELECTION_COUNT:
        raise ValueError("stage-two source dataset IDs are not unique")
    if len({spec.local_id for spec in SPECS}) != SELECTION_COUNT:
        raise ValueError("stage-two local dataset IDs are not unique")

    stage1_bundle = json.loads(stage1_bundle_path.read_text(encoding="utf-8"))
    stage1_source_ids = {
        str(row["source_dataset_id"]) for row in stage1_bundle["scenarios"]
    }
    if stage1_source_ids & {spec.dataset_id for spec in SPECS}:
        raise ValueError("stage-two source IDs overlap stage one")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_rows = {row["dataset_id"]: row for row in audit["datasets"]}
    latest = _latest_remote_revisions(remote_listing)
    prior = _prior_sources(
        stage1_split_path=stage1_split_path,
        old_design_path=old_design_path,
        historical_analysis_path=historical_analysis_path,
    )

    encoders: dict[Path, TrainingEncoder] = {}
    processor_contract = {
        "schema": "sft_qwen3_text_processor_contract/v3",
        "template": TEMPLATE,
        "padding": "dynamic_pad_to_multiple_of_8",
        "profile_row_fields": [
            "sample_id",
            "total_tokens",
            "label_tokens",
            "turns",
            "assistant_turns",
        ],
        "models": {
            spec.model_id: _tokenizer_binding(spec.model_path) for spec in SPECS
        },
    }
    write_json(PROCESSOR_CONTRACT, processor_contract)

    source_rows: dict[str, list[dict[str, str]]] = {}
    source_fingerprints: dict[str, set[str]] = {}
    prior_overlap_rows: list[dict[str, Any]] = []
    for spec in SPECS:
        audit_row = audit_rows.get(spec.dataset_id)
        if not audit_row or audit_row.get("accepted_pure_text_sft") is not True:
            raise ValueError(f"source did not pass text-SFT audit: {spec.dataset_id}")
        latest_row = latest.get(spec.dataset_id)
        if not latest_row or latest_row["remote_key"] != audit_row["remote_key"]:
            raise ValueError(f"source is not latest published revision: {spec.dataset_id}")
        path = Path(audit_row["local_path"])
        normalized: dict[str, dict[str, str]] = {}
        for value in _iter_objects(path):
            row = _normalize(value, path=path)
            normalized.setdefault(_fingerprint(row), row)
        if len(normalized) < 100:
            raise ValueError(f"source has fewer than 100 unique rows: {spec.dataset_id}")
        fingerprints = set(normalized)
        for prior_id, prior_fingerprints in prior.items():
            shared = len(fingerprints & prior_fingerprints)
            if shared:
                prior_overlap_rows.append(
                    {
                        "stage2_dataset_id": spec.dataset_id,
                        "prior_source_id": prior_id,
                        "shared_rows": shared,
                    }
                )
        source_rows[spec.dataset_id] = list(normalized.values())
        source_fingerprints[spec.dataset_id] = fingerprints
    if prior_overlap_rows:
        raise ValueError(f"stage-two sources overlap prior evidence: {prior_overlap_rows}")

    selected_overlap_rows: list[dict[str, Any]] = []
    for index, left in enumerate(SPECS):
        for right in SPECS[index + 1 :]:
            shared = len(
                source_fingerprints[left.dataset_id]
                & source_fingerprints[right.dataset_id]
            )
            if shared:
                selected_overlap_rows.append(
                    {
                        "left_dataset_id": left.dataset_id,
                        "right_dataset_id": right.dataset_id,
                        "shared_rows": shared,
                    }
                )
    if selected_overlap_rows:
        raise ValueError(
            f"stage-two selected sources are not exact-row disjoint: {selected_overlap_rows}"
        )

    scenarios: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for spec in SPECS:
        if spec.model_path not in encoders:
            encoders[spec.model_path] = TrainingEncoder(spec.model_path)
        encoder = encoders[spec.model_path]
        encoded: list[dict[str, Any]] = []
        for row in source_rows[spec.dataset_id]:
            fingerprint = _fingerprint(row)
            total_tokens, label_tokens = encoder.encode(row)
            encoded.append(
                {
                    "fingerprint": fingerprint,
                    "row": row,
                    "total_tokens": int(total_tokens),
                    "label_tokens": int(label_tokens),
                }
            )
        encoded.sort(key=lambda item: item["fingerprint"])
        longest = max(
            encoded,
            key=lambda item: (item["total_tokens"], item["fingerprint"]),
        )
        selected = encoded[:MAX_SELECTED_ROWS]
        if longest not in selected:
            selected[-1] = longest
            selected.sort(key=lambda item: item["fingerprint"])

        data_rows: list[dict[str, Any]] = []
        profile_rows: list[dict[str, Any]] = []
        for item in selected:
            sample_id = f"{spec.local_id}:source_fp:{item['fingerprint'][:20]}"
            data_rows.append({"sample_id": sample_id, **item["row"]})
            profile_rows.append(
                {
                    "sample_id": sample_id,
                    "total_tokens": item["total_tokens"],
                    "label_tokens": item["label_tokens"],
                    "turns": 3 if item["row"]["system"] else 2,
                    "assistant_turns": 1,
                }
            )
        data_path = DATA_OUTPUT_DIR / f"{spec.local_id}.jsonl"
        profile_path = PROFILE_DIR / f"{spec.local_id}.{TEMPLATE}.jsonl"
        _atomic_jsonl(data_path, data_rows)
        _atomic_jsonl(profile_path, profile_rows)

        lengths = [row["total_tokens"] for row in profile_rows]
        raw_max = max(lengths)
        ratio = raw_max / spec.cutoff_len
        actual_bin = _ratio_bin(ratio)
        if actual_bin != spec.ratio_bin:
            raise ValueError(
                f"ratio-bin drift for {spec.local_id}: expected {spec.ratio_bin}, "
                f"got {actual_bin} from {raw_max}/{spec.cutoff_len}"
            )
        aligned_effective = min(spec.cutoff_len, _round_up(raw_max, 8))
        audit_row = audit_rows[spec.dataset_id]
        source_path = Path(audit_row["local_path"])
        split_rows.append(
            {
                "dataset_id": spec.local_id,
                "source_dataset_id": spec.dataset_id,
                "source_remote_key": audit_row["remote_key"],
                "source_revision": latest[spec.dataset_id]["revision"],
                "source_path": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "source_unique_rows": len(encoded),
                "selected_rows": len(selected),
                "selected_fingerprints_sha256": sha256_json(
                    [item["fingerprint"] for item in selected]
                ),
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
            }
        )
        scenarios.append(
            {
                "scenario_id": (
                    f"{spec.local_id}__{spec.model_id}_cutoff{spec.cutoff_len}"
                ),
                "split_unit_id": spec.dataset_id,
                "dataset_id": spec.local_id,
                "source_dataset_id": spec.dataset_id,
                "business_scene": spec.business_scene,
                "dataset_category": "longtail" if raw_max > 2048 else "short",
                "model_id": spec.model_id,
                "model_path": str(spec.model_path.resolve()),
                "train_type": "lora",
                "cutoff_len": spec.cutoff_len,
                "target_gbs": 64,
                "ratio_bin": spec.ratio_bin,
                "raw_profile_max": raw_max,
                "raw_profile_max_over_cutoff": ratio,
                "aligned_effective_sequence": aligned_effective,
                "effective_sequence_over_cutoff": (
                    aligned_effective / spec.cutoff_len
                ),
                "profile_statistics": {
                    "rows": len(lengths),
                    "minimum": min(lengths),
                    "mean": sum(lengths) / len(lengths),
                    "p50": percentile(lengths, 50),
                    "p90": percentile(lengths, 90),
                    "p95": percentile(lengths, 95),
                    "p99": percentile(lengths, 99),
                    "maximum": raw_max,
                },
                "data_path": str(data_path.resolve()),
                "data_sha256": sha256_file(data_path),
                "profile_path": str(profile_path.resolve()),
                "profile_sha256": sha256_file(profile_path),
                "matrix_roles": {"stage2_critical_lora": True},
            }
        )

    bin_counts = Counter(row["ratio_bin"] for row in scenarios)
    expected_bins = {
        "le_0p40": 5,
        "gt_0p40_le_0p70": 5,
        "gt_0p70_le_0p90": 5,
        "gt_0p95": 5,
    }
    if dict(bin_counts) != expected_bins:
        raise ValueError(f"stage-two ratio bins are unbalanced: {dict(bin_counts)}")
    model_counts = Counter(row["model_id"] for row in scenarios)
    if dict(model_counts) != {"qwen3_4b": 7, "qwen3_8b": 6, "qwen3_14b": 7}:
        raise ValueError(f"stage-two model counts drifted: {dict(model_counts)}")

    split_manifest = {
        "schema": "sft_h800_lora_safety_stage2_split_manifest/v1",
        "remote_bucket": REMOTE_BUCKET,
        "remote_prefix": REMOTE_PREFIX,
        "remote_listing": {
            "path_at_selection": str(remote_listing.resolve()),
            "sha256": sha256_file(remote_listing),
            "line_count": sum(1 for _ in remote_listing.open(encoding="utf-8")),
        },
        "candidate_audit": {
            "path": str(audit_path.resolve()),
            "sha256": sha256_file(audit_path),
        },
        "selection_rows": split_rows,
        "selected_cross_source_exact_overlaps": selected_overlap_rows,
        "selected_vs_all_prior_exact_overlaps": prior_overlap_rows,
        "prior_source_groups_checked": len(prior),
    }
    write_json(SPLIT_MANIFEST, split_manifest)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "execution_authorized": False,
        "remote_bucket": REMOTE_BUCKET,
        "remote_prefix": REMOTE_PREFIX,
        "selection_contract": {
            "new_source_dataset_ids_vs_stage1": True,
            "latest_published_revision_required": True,
            "pure_text_system_prompt_response_required": True,
            "zero_exact_normalized_row_overlap_with_all_prior_sources": True,
            "zero_exact_normalized_row_overlap_between_stage2_sources": True,
            "maximum_rows_per_frozen_scenario": MAX_SELECTED_ROWS,
            "global_longest_row_forced_into_frozen_slice": True,
            "tokenizer_template_preprocessing_bound_before_gpu": True,
        },
        "counts": {
            "scenarios": len(scenarios),
            "ratio_bins": dict(bin_counts),
            "models": dict(model_counts),
            "jobs_planned": 60,
            "gpu_job_equivalents": 120,
        },
        "bindings": {
            "candidate_audit": {
                "path": str(audit_path.resolve()),
                "sha256": sha256_file(audit_path),
            },
            "stage1_bundle": {
                "path": str(stage1_bundle_path.resolve()),
                "sha256": sha256_file(stage1_bundle_path),
            },
            "stage1_split": {
                "path": str(stage1_split_path.resolve()),
                "sha256": sha256_file(stage1_split_path),
            },
            "old_calibration_design": {
                "path": str(old_design_path.resolve()),
                "sha256": sha256_file(old_design_path),
            },
            "historical_dataset_analysis": {
                "path": str(historical_analysis_path.resolve()),
                "sha256": sha256_file(historical_analysis_path),
            },
            "processor_contract": {
                "path": str(PROCESSOR_CONTRACT.resolve()),
                "sha256": sha256_file(PROCESSOR_CONTRACT),
            },
            "split_manifest": {
                "path": str(SPLIT_MANIFEST.resolve()),
                "sha256": sha256_file(SPLIT_MANIFEST),
            },
        },
        "scenarios": scenarios,
    }
    report["report_sha256"] = sha256_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--remote-listing", type=Path, default=DEFAULT_REMOTE_LISTING)
    parser.add_argument("--stage1-bundle", type=Path, default=DEFAULT_STAGE1_BUNDLE)
    parser.add_argument("--stage1-split", type=Path, default=DEFAULT_STAGE1_SPLIT)
    parser.add_argument("--old-design", type=Path, default=DEFAULT_OLD_DESIGN)
    parser.add_argument(
        "--historical-analysis", type=Path, default=DEFAULT_HISTORICAL_ANALYSIS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    outputs = (args.output, PROCESSOR_CONTRACT, SPLIT_MANIFEST)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing stage-two data: {existing}")
    report = prepare(
        audit_path=args.audit,
        remote_listing=args.remote_listing,
        stage1_bundle_path=args.stage1_bundle,
        stage1_split_path=args.stage1_split,
        old_design_path=args.old_design,
        historical_analysis_path=args.historical_analysis,
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scenarios": report["counts"]["scenarios"],
                "ratio_bins": report["counts"]["ratio_bins"],
                "models": report["counts"]["models"],
                "gpu_training_started": False,
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
