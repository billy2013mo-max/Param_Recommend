#!/usr/bin/env python3
"""Freeze 15 source-disjoint SFT datasets and exact Qwen3 token profiles.

This is a CPU-only preparation step.  The input files were downloaded from
``infra-ai-infra-storage/datasets/`` and audited before selection.  The script
rejects stale published revisions, non-text SFT schemas, row overlap between
selected source IDs, and overlap with the four earlier calibration datasets.
It never launches a GPU process.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

from transformers import AutoTokenizer
from llamafactory.data.template import TEMPLATES

from common import ARTIFACT_DIR, DATA_DIR, ROOT, percentile, sha256_file, sha256_json, write_json


SCHEMA = "sft_h800_lora_source_disjoint_data_bundle/v1"
SELECTION_COUNT = 15
MAX_SELECTED_ROWS = 768
TEMPLATE = "qwen3_nothink"
REMOTE_BUCKET = "infra-ai-infra-storage"
REMOTE_PREFIX = "datasets/"
DEFAULT_AUDIT = ARTIFACT_DIR / "h800_lora_remote_candidate_audit_v1.json"
DEFAULT_REMOTE_LISTING = Path("/tmp/ai_infra_storage_datasets.tsv")
DEFAULT_OLD_DESIGN = ARTIFACT_DIR / "h800_profile_aware_memory_calibration_design_v1.json"
DEFAULT_OUTPUT = ARTIFACT_DIR / "h800_lora_source_disjoint_bundle_v1.json"
DATA_OUTPUT_DIR = DATA_DIR / "h800_lora_source_disjoint_v1"
PROFILE_DIR = ARTIFACT_DIR / "h800_lora_source_disjoint_v1" / "profiles"
PROCESSOR_CONTRACT = ARTIFACT_DIR / "h800_lora_source_disjoint_v1" / "processor_contract.json"
SPLIT_MANIFEST = ARTIFACT_DIR / "h800_lora_source_disjoint_v1" / "split_manifest.json"
LISTING_PATTERN = re.compile(
    r"^(?P<size>\d+)\t(?P<key>datasets/(?P<dataset_id>dataset-[^/]+)/"
    r"(?P<revision>\d+)/publish/[^/]+\.jsonl)$"
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    local_id: str
    model_id: str
    model_path: Path
    cutoff_len: int
    ratio_bin: str
    business_scene: str
    a2_full_control: bool = False
    a3_four_gpu_pair: bool = False
    a4_repeat_control: bool = False


SPECS = (
    DatasetSpec(
        "dataset-36recb-1780063136", "lora_src01_comment_safety_short",
        "qwen3_4b", Path("/wanqing-models/Qwen3-4B"), 2048, "le_0p40",
        "评论不友善内容审核", True, True, True,
    ),
    DatasetSpec(
        "dataset-vad2dy-1783330395", "lora_src02_live_script_polish",
        "qwen3_8b", Path("/wanqing-models/Qwen3-8B"), 2048, "le_0p40",
        "直播话术合并润色",
    ),
    DatasetSpec(
        "dataset-w0i17d-1783493358", "lora_src03_news_summary_short",
        "qwen3_14b", Path("/wanqing-models/Qwen3-14B"), 2048, "le_0p40",
        "新闻摘要生成", True,
    ),
    DatasetSpec(
        "dataset-vx2vvb-1785828598", "lora_src04_account_risk_short",
        "qwen3_4b", Path("/wanqing-models/Qwen3-4B"), 2048, "le_0p40",
        "账号导流与私单风险识别",
    ),
    DatasetSpec(
        "dataset-64de8r-1785128542", "lora_src05_collection_record",
        "qwen3_4b", Path("/wanqing-models/Qwen3-4B"), 4096, "gt_0p40_le_0p70",
        "催收通话记录整理", True,
    ),
    DatasetSpec(
        "dataset-6j8f7g-1785469920", "lora_src06_video_comment_audit",
        "qwen3_8b", Path("/wanqing-models/Qwen3-8B"), 4096, "gt_0p40_le_0p70",
        "短视频评论安全审核", False, True, True,
    ),
    DatasetSpec(
        "dataset-grkvbh-1784610336", "lora_src07_flood_event_classify",
        "qwen3_8b", Path("/wanqing-models/Qwen3-8B"), 8192, "gt_0p40_le_0p70",
        "汛期热点事件分类",
    ),
    DatasetSpec(
        "dataset-t7o109-1784029430", "lora_src08_ecommerce_video_structure",
        "qwen3_14b", Path("/wanqing-models/Qwen3-14B"), 8192, "gt_0p40_le_0p70",
        "电商视频结构化分析", True,
    ),
    DatasetSpec(
        "dataset-2kt76w-1780898786", "lora_src09_employment_type",
        "qwen3_4b", Path("/wanqing-models/Qwen3-4B"), 2048, "gt_0p70_le_0p90",
        "候选人用工性质分类",
    ),
    DatasetSpec(
        "dataset-p5uuj1-1785392941", "lora_src10_medical_product_name",
        "qwen3_8b", Path("/wanqing-models/Qwen3-8B"), 2048, "gt_0p70_le_0p90",
        "消费医疗商品名称校验", True,
    ),
    DatasetSpec(
        "dataset-sn4fof-1785220077", "lora_src11_brand_consistency",
        "qwen3_14b", Path("/wanqing-models/Qwen3-14B"), 4096, "gt_0p70_le_0p90",
        "电商商品品牌一致性审核",
    ),
    DatasetSpec(
        "dataset-h1bt3r-1784719136", "lora_src12_marketing_antifraud_long",
        "qwen3_14b", Path("/wanqing-models/Qwen3-14B"), 16384, "gt_0p70_le_0p90",
        "电商营销点击行为反作弊", True, True, True,
    ),
    DatasetSpec(
        "dataset-p7f109-1783932688", "lora_src13_private_trade_compliance",
        "qwen3_4b", Path("/wanqing-models/Qwen3-4B"), 8192, "gt_0p95",
        "本地生活私下交易合规审核", True, True, True,
    ),
    DatasetSpec(
        "dataset-jcefdi-1785404432", "lora_src14_ad_audience_match",
        "qwen3_8b", Path("/wanqing-models/Qwen3-8B"), 4096, "gt_0p95",
        "广告人群包匹配", True,
    ),
    DatasetSpec(
        "dataset-tja6rp-1785405106", "lora_src15_live_highlight_long",
        "qwen3_14b", Path("/wanqing-models/Qwen3-14B"), 8192, "gt_0p95",
        "直播高光片段提取",
    ),
)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _iter_objects(path: Path) -> Iterable[Any]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, list):
                yield from value
            else:
                yield value


def _normalize(value: Any, *, path: Path) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} contains a non-object row")
    required = ("system", "prompt", "response")
    if not all(isinstance(value.get(field), str) for field in required):
        raise ValueError(f"{path} contains a row without string system/prompt/response")
    return {field: value[field] for field in required}


def _fingerprint(row: dict[str, str]) -> str:
    normalized = "\0".join(row[field].strip() for field in ("system", "prompt", "response"))
    return hashlib.sha256(normalized.encode()).hexdigest()


class TrainingEncoder:
    def __init__(self, model_path: Path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        self.template = copy.deepcopy(TEMPLATES[TEMPLATE])
        self.template.fix_special_tokens(self.tokenizer)

    def encode(self, row: dict[str, str]) -> tuple[int, int]:
        conversation = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
        pairs = self.template.encode_multiturn(
            self.tokenizer, conversation, system=row["system"] or None, tools=None
        )
        source_tokens = sum(len(source_ids) for source_ids, _ in pairs)
        label_tokens = sum(len(target_ids) for _, target_ids in pairs)
        if self.template.efficient_eos:
            label_tokens += 1
        return source_tokens + label_tokens, label_tokens


def _tokenizer_binding(path: Path) -> dict[str, Any]:
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    files = {
        name: sha256_file(path / name)
        for name in names
        if (path / name).is_file()
    }
    return {
        "path": str(path.resolve()),
        "template": TEMPLATE,
        "files": files,
        "files_sha256": sha256_json(files),
    }


def _latest_remote_revisions(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LISTING_PATTERN.match(line)
        if not match:
            continue
        row = {
            "remote_key": match.group("key"),
            "revision": int(match.group("revision")),
            "file_size_bytes": int(match.group("size")),
        }
        dataset_id = match.group("dataset_id")
        if row["revision"] > latest.get(dataset_id, {}).get("revision", -1):
            latest[dataset_id] = row
    return latest


def _ratio_bin(ratio: float) -> str:
    if ratio <= 0.40:
        return "le_0p40"
    if ratio <= 0.70:
        return "gt_0p40_le_0p70"
    if ratio <= 0.90:
        return "gt_0p70_le_0p90"
    if ratio > 0.95:
        return "gt_0p95"
    return "forbidden_gap_gt_0p90_le_0p95"


def _round_up(value: int, multiple: int = 8) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def prepare(
    *, audit_path: Path, remote_listing: Path, old_design_path: Path
) -> dict[str, Any]:
    if len(SPECS) != SELECTION_COUNT:
        raise ValueError(f"expected {SELECTION_COUNT} specs, got {len(SPECS)}")
    if len({spec.dataset_id for spec in SPECS}) != SELECTION_COUNT:
        raise ValueError("selected remote dataset IDs are not unique")
    if len({spec.local_id for spec in SPECS}) != SELECTION_COUNT:
        raise ValueError("selected local dataset IDs are not unique")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_rows = {row["dataset_id"]: row for row in audit["datasets"]}
    latest = _latest_remote_revisions(remote_listing)
    old_design = json.loads(old_design_path.read_text(encoding="utf-8"))
    old_fingerprints: dict[str, set[str]] = {}
    for scenario in old_design["scenarios"]:
        old_fingerprints[scenario["dataset_id"]] = {
            _fingerprint(_normalize(value, path=Path(scenario["data_path"])))
            for value in _iter_objects(Path(scenario["data_path"]))
        }

    encoders: dict[Path, TrainingEncoder] = {}
    processor_contract = {
        "schema": "sft_qwen3_text_processor_contract/v3",
        "template": TEMPLATE,
        "padding": "dynamic_pad_to_multiple_of_8",
        "profile_row_fields": [
            "sample_id", "total_tokens", "label_tokens", "turns", "assistant_turns"
        ],
        "models": {
            spec.model_id: _tokenizer_binding(spec.model_path) for spec in SPECS
        },
    }
    write_json(PROCESSOR_CONTRACT, processor_contract)

    source_fingerprints: dict[str, set[str]] = {}
    source_rows_by_id: dict[str, list[dict[str, str]]] = {}
    for spec in SPECS:
        audit_row = audit_rows.get(spec.dataset_id)
        if not audit_row or audit_row.get("accepted_pure_text_sft") is not True:
            raise ValueError(f"selected source did not pass the SFT audit: {spec.dataset_id}")
        latest_row = latest.get(spec.dataset_id)
        if not latest_row or latest_row["remote_key"] != audit_row["remote_key"]:
            raise ValueError(
                f"selected source is not the latest published revision: {spec.dataset_id}"
            )
        source_path = Path(audit_row["local_path"])
        rows: dict[str, dict[str, str]] = {}
        for value in _iter_objects(source_path):
            row = _normalize(value, path=source_path)
            rows.setdefault(_fingerprint(row), row)
        if len(rows) < 100:
            raise ValueError(f"selected source has fewer than 100 unique rows: {spec.dataset_id}")
        source_fingerprints[spec.dataset_id] = set(rows)
        source_rows_by_id[spec.dataset_id] = list(rows.values())

    selected_overlap: list[dict[str, Any]] = []
    for index, left in enumerate(SPECS):
        left_set = source_fingerprints[left.dataset_id]
        for right in SPECS[index + 1:]:
            right_set = source_fingerprints[right.dataset_id]
            shared = len(left_set & right_set)
            if shared:
                row = {
                    "left_dataset_id": left.dataset_id,
                    "right_dataset_id": right.dataset_id,
                    "shared_rows": shared,
                    "left_containment": shared / len(left_set),
                    "right_containment": shared / len(right_set),
                }
                selected_overlap.append(row)
                if max(row["left_containment"], row["right_containment"]) >= 0.05:
                    raise ValueError(f"selected sources violate the 5% overlap limit: {row}")

    old_overlap: list[dict[str, Any]] = []
    for spec in SPECS:
        current = source_fingerprints[spec.dataset_id]
        for old_id, previous in old_fingerprints.items():
            shared = len(current & previous)
            if shared:
                old_overlap.append(
                    {
                        "dataset_id": spec.dataset_id,
                        "old_split_unit_id": old_id,
                        "shared_rows": shared,
                        "new_containment": shared / len(current),
                        "old_containment": shared / len(previous),
                    }
                )
                if max(shared / len(current), shared / len(previous)) >= 0.05:
                    raise ValueError(
                        f"selected source overlaps the old calibration split: {spec.dataset_id}"
                    )

    scenarios: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for spec in SPECS:
        if spec.model_path not in encoders:
            encoders[spec.model_path] = TrainingEncoder(spec.model_path)
        encoder = encoders[spec.model_path]
        encoded: list[dict[str, Any]] = []
        for row in source_rows_by_id[spec.dataset_id]:
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
        longest = max(encoded, key=lambda item: (item["total_tokens"], item["fingerprint"]))
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
                f"ratio bin drift for {spec.local_id}: expected {spec.ratio_bin}, "
                f"got {actual_bin} from {raw_max}/{spec.cutoff_len}"
            )
        aligned_effective = min(spec.cutoff_len, _round_up(raw_max, 8))
        audit_row = audit_rows[spec.dataset_id]
        source_path = Path(audit_row["local_path"])
        latest_row = latest[spec.dataset_id]
        split_rows.append(
            {
                "dataset_id": spec.local_id,
                "source_dataset_id": spec.dataset_id,
                "source_remote_key": audit_row["remote_key"],
                "source_revision": latest_row["revision"],
                "source_file_size_bytes": latest_row["file_size_bytes"],
                "source_path": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "source_unique_rows": len(encoded),
                "selection_policy": "lowest_sha256_fingerprints_plus_global_longest_v1",
                "maximum_selected_rows": MAX_SELECTED_ROWS,
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
                "scenario_id": f"{spec.local_id}__{spec.model_id}_cutoff{spec.cutoff_len}",
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
                "effective_sequence_over_cutoff": aligned_effective / spec.cutoff_len,
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
                "matrix_roles": {
                    "a1_critical_lora": True,
                    "a2_full_control": spec.a2_full_control,
                    "a3_four_gpu_pair": spec.a3_four_gpu_pair,
                    "a4_repeat_control": spec.a4_repeat_control,
                },
            }
        )

    bin_counts = Counter(row["ratio_bin"] for row in scenarios)
    expected_counts = {
        "le_0p40": 4,
        "gt_0p40_le_0p70": 4,
        "gt_0p70_le_0p90": 4,
        "gt_0p95": 3,
    }
    if dict(bin_counts) != expected_counts:
        raise ValueError(f"ratio-bin count mismatch: {dict(bin_counts)}")
    model_counts = Counter(row["model_id"] for row in scenarios)
    if dict(model_counts) != {"qwen3_4b": 5, "qwen3_8b": 5, "qwen3_14b": 5}:
        raise ValueError(f"model count mismatch: {dict(model_counts)}")
    role_counts = {
        role: sum(bool(row["matrix_roles"][role]) for row in scenarios)
        for role in ("a1_critical_lora", "a2_full_control", "a3_four_gpu_pair", "a4_repeat_control")
    }
    if role_counts != {
        "a1_critical_lora": 15,
        "a2_full_control": 8,
        "a3_four_gpu_pair": 4,
        "a4_repeat_control": 4,
    }:
        raise ValueError(f"matrix role count mismatch: {role_counts}")

    split_manifest = {
        "schema": "sft_h800_lora_source_disjoint_split_manifest/v1",
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
        "selected_cross_source_exact_overlaps": selected_overlap,
        "selected_vs_old_calibration_exact_overlaps": old_overlap,
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
            "source_dataset_ids_unique": True,
            "latest_published_revision_required": True,
            "pure_text_system_prompt_response_required": True,
            "maximum_cross_source_exact_row_containment": 0.05,
            "maximum_rows_per_frozen_scenario": MAX_SELECTED_ROWS,
            "global_longest_row_forced_into_frozen_slice": True,
            "tokenizer_template_preprocessing_bound_before_gpu": True,
        },
        "counts": {
            "scenarios": len(scenarios),
            "ratio_bins": dict(bin_counts),
            "models": dict(model_counts),
            "matrix_roles": role_counts,
            "first_stage_jobs_planned": 77,
        },
        "bindings": {
            "candidate_audit": {
                "path": str(audit_path.resolve()), "sha256": sha256_file(audit_path)
            },
            "old_calibration_design": {
                "path": str(old_design_path.resolve()), "sha256": sha256_file(old_design_path)
            },
            "processor_contract": {
                "path": str(PROCESSOR_CONTRACT.resolve()), "sha256": sha256_file(PROCESSOR_CONTRACT)
            },
            "split_manifest": {
                "path": str(SPLIT_MANIFEST.resolve()), "sha256": sha256_file(SPLIT_MANIFEST)
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
    parser.add_argument("--old-design", type=Path, default=DEFAULT_OLD_DESIGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = prepare(
        audit_path=args.audit,
        remote_listing=args.remote_listing,
        old_design_path=args.old_design,
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scenarios": report["counts"]["scenarios"],
                "ratio_bins": report["counts"]["ratio_bins"],
                "models": report["counts"]["models"],
                "gpu_training_started": report["gpu_training_started"],
                "report_sha256": report["report_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
