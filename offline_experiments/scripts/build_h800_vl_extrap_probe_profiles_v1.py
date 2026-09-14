#!/usr/bin/env python3
"""给外推验证批补数据集画像（不碰数据集本身）。

为什么单独写一个脚本
--------------------
现成的 build_h800_vl_visual_share_sweep_v1.py 会连数据集 jsonl 一起重新生成。
那些文件的 sha256 正被已冻结的队列引用，重写会让在跑/已跑的作业对不上账。
本脚本只读数据集、只写画像。

与现有画像同口径（已校准）
--------------------------
用 qwen2p5_vl_3b.f1.short.low 现有画像反推验证过：

    text_tokens  = tokenize(user 全文，含 <image> 标签)，不加特殊 token  -> 49 ✓
    label_tokens = tokenize(assistant 全文)                          -> 27 ✓

区别在于本脚本直接读数据集里的**实际文本**，而原脚本是各模型各自「撑」一版
指令再数自己那版。数据集只有一份（第一个模型撑出来的那版），所以原口径下
后续模型的 text_tokens 严格说不对应数据集实际内容。本脚本没有这个问题。

视觉 token 用同一套确定性 smart_resize 重算，不加载 processor。
样本循环复用 999 个图片组，所以按图片几何做了缓存。

只跑 CPU。不建审批，不启动 GPU。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from common import ARTIFACT_DIR, DATA_DIR, write_json, sha256_file

PROFILE_DIR = ARTIFACT_DIR / "h800_vl_visual_share_sweep_profiles_v1"
OUT_DIR = DATA_DIR / "vl_visual_share_sweep_v1"
MANIFEST = ARTIFACT_DIR / "h800_vl_extrap_probe_profiles_manifest_v1.json"
CHANNELS = 3

# 三个长度档，跨度 16.3 倍（中心/吞吐门槛要 >=3 倍）。
# 实测长度中位：286 / 1881 / 4674
CELLS = (
    {"frames": 1, "text_tier": "short", "media_tier": "low"},
    {"frames": 2, "text_tier": "medium", "media_tier": "high"},
    {"frames": 4, "text_tier": "long", "media_tier": "high"},
)

TIER_PIXELS = {
    "low": {"image_min_pixels": 3136, "image_max_pixels": 200704},
    "high": {"image_min_pixels": 3136, "image_max_pixels": 589824},
}

# 路径来自本地权重目录；patch/merge 一律从 preprocessor_config.json 实测，不硬编码
MODELS = {
    "qwen3p5_0p8b": "/wanqing-models/Qwen3.5-0.8B",
    "qwen3p5_9b": "/wanqing-models/Qwen3.5-9B",
    "qwen3p5_27b": "/wanqing-models/Qwen3.5-27B",
    "qwen2p5_vl_7b": "/wanqing-models/Qwen2.5-VL-7B-Instruct",
    "qwen3_vl_8b": "/wanqing-models/Qwen3-VL-8B-Instruct",
    "qwen2p5_vl_32b": "/wanqing-models/Qwen2.5-VL-32B-Instruct",
}


def processor_geometry(root: str) -> dict:
    config = json.loads(Path(root, "preprocessor_config.json").read_text())
    patch = config.get("patch_size")
    merge = config.get("merge_size") or config.get("spatial_merge_size")
    temporal = config.get("temporal_patch_size") or 1
    if not patch or not merge:
        raise ValueError("%s 缺 patch_size/merge_size" % root)
    return {"patch_size": int(patch), "spatial_merge_size": int(merge),
            "temporal_patch_size": int(temporal), "path": root}


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Qwen2-VL 的确定性 smart_resize，与 build_h800_vl_visual_share_sweep_v1 一致。"""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


_size_cache: dict[str, tuple[int, int]] = {}
_record_cache: dict[tuple, dict] = {}


def image_record(path: str, processor: dict, tier: dict) -> dict:
    key = (path, processor["patch_size"], processor["spatial_merge_size"],
           processor["temporal_patch_size"], tier["image_min_pixels"],
           tier["image_max_pixels"])
    hit = _record_cache.get(key)
    if hit is not None:
        return hit
    size = _size_cache.get(path)
    if size is None:
        with Image.open(path) as image:
            size = image.size
        _size_cache[path] = size
    width, height = size
    patch = processor["patch_size"]
    merge = processor["spatial_merge_size"]
    temporal = processor["temporal_patch_size"]
    factor = patch * merge
    resized_h, resized_w = smart_resize(
        height, width, factor, tier["image_min_pixels"], tier["image_max_pixels"])
    grid_h, grid_w = resized_h // patch, resized_w // patch
    raw = grid_h * grid_w
    record = {
        "path": path,
        "original_size": [width, height],
        "resized_size": [resized_w, resized_h],
        "grid_thw": [1, grid_h, grid_w],
        "raw_patch_units": raw,
        "visual_tokens": raw // (merge * merge),
        "pixel_values_elements": raw * CHANNELS * temporal * patch * patch,
    }
    _record_cache[key] = record
    return record


def distribution(records, key):
    values = sorted(r[key] for r in records)
    n = len(values)
    pick = lambda q: values[min(n - 1, int(q * n))]
    return {"min": values[0], "mean": sum(values) / n, "p50": pick(0.5),
            "p90": pick(0.9), "p95": pick(0.95), "p99": pick(0.99),
            "max": values[-1]}


def main() -> None:
    from transformers import AutoTokenizer

    written = []
    for model_id, root in MODELS.items():
        processor = processor_geometry(root)
        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
        for cell in CELLS:
            frames = cell["frames"]
            text_tier = cell["text_tier"]
            media_tier = cell["media_tier"]
            tier = TIER_PIXELS[media_tier]
            data_path = OUT_DIR / ("f%d" % frames) / (
                "vl_share_f%d_%s.jsonl" % (frames, text_tier))
            if not data_path.is_file():
                raise FileNotFoundError(data_path)

            records = []
            # 同一档位的所有行文本完全相同（只有图片不同），tokenize 一次即可。
            # 已验证：现有画像里 text_tokens 的 min=p50=max，印证了这一点。
            text_cache: dict[str, int] = {}

            def count_tokens(text: str) -> int:
                hit = text_cache.get(text)
                if hit is None:
                    hit = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    text_cache[text] = hit
                return hit

            with data_path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    msgs = {m["role"]: m["content"] for m in row["messages"]}
                    text_tokens = count_tokens(msgs["user"])
                    label_tokens = count_tokens(msgs["assistant"])
                    image_records = [image_record(p, processor, tier)
                                     for p in row["images"]]
                    visual = sum(r["visual_tokens"] for r in image_records)
                    records.append({
                        "sample_id": str(index),
                        "task_family": "vl_visual_share_sweep",
                        "image_count": len(image_records),
                        "images": image_records,
                        "text_tokens": text_tokens,
                        "label_tokens": label_tokens,
                        "visual_tokens_total": visual,
                        "raw_patch_units_total": sum(
                            r["raw_patch_units"] for r in image_records),
                        "pixel_values_elements_total": sum(
                            r["pixel_values_elements"] for r in image_records),
                        "total_tokens": text_tokens + label_tokens + visual,
                    })

            profile = {
                "schema": "sft_h800_vl_visual_share_profile/v1",
                "model": {"id": model_id, "path": root},
                "text_tier": text_tier,
                "media_tier": media_tier,
                "frames": frames,
                "processor_binding": {
                    "patch_size": processor["patch_size"],
                    "spatial_merge_size": processor["spatial_merge_size"],
                    "temporal_patch_size": processor["temporal_patch_size"],
                    "input_channels": CHANNELS,
                    **tier,
                    "grid_contract": "image_grid_thw_after_smart_resize",
                    "reproduction": "deterministic_qwen2vl_smart_resize_no_processor_load",
                },
                "text_tokens_measured_with_tokenizer": True,
                "text_tokens_source": "dataset_actual_message_text",
                "records": records,
                "summary": {
                    "records": len(records),
                    "images_per_sample": distribution(records, "image_count"),
                    "text_tokens": distribution(records, "text_tokens"),
                    "label_tokens": distribution(records, "label_tokens"),
                    "visual_tokens_total": distribution(records, "visual_tokens_total"),
                    "raw_patch_units_total": distribution(records, "raw_patch_units_total"),
                    "pixel_values_elements_total": distribution(
                        records, "pixel_values_elements_total"),
                    "total_tokens": distribution(records, "total_tokens"),
                },
            }
            path = PROFILE_DIR / ("%s.f%d.%s.%s.json" % (
                model_id, frames, text_tier, media_tier))
            write_json(path, profile)
            mean_visual = profile["summary"]["visual_tokens_total"]["mean"]
            mean_total = profile["summary"]["total_tokens"]["mean"]
            written.append({
                "model_id": model_id, "frames": frames, "text_tier": text_tier,
                "media_tier": media_tier, "path": str(path),
                "sha256": sha256_file(path),
                "data_path": str(data_path), "data_sha256": sha256_file(data_path),
                "records": len(records),
                "text_tokens_p50": profile["summary"]["text_tokens"]["p50"],
                "total_tokens_p50": profile["summary"]["total_tokens"]["p50"],
                "visual_share": mean_visual / mean_total,
            })
            print("%-16s f%d %-7s %-5s  文本 %5d  总长 %6.0f  视觉占比 %.2f" % (
                model_id, frames, text_tier, media_tier,
                profile["summary"]["text_tokens"]["p50"], mean_total,
                mean_visual / mean_total))

    write_json(MANIFEST, {
        "schema": "sft_h800_vl_extrap_probe_profiles_manifest/v1",
        "purpose": "外推验证批的画像补充：6 个模型 x 3 个长度档",
        "token_convention": {
            "text_tokens": "tokenize(user 全文，含 <image>)，不加特殊 token",
            "label_tokens": "tokenize(assistant 全文)",
            "calibrated_against": "qwen2p5_vl_3b.f1.short.low 现有画像 (49/27)",
            "difference_from_original": "本脚本读数据集实际文本；原脚本各模型各自撑指令",
        },
        "dataset_untouched": True,
        "profiles": written,
    })
    print("")
    print("画像 %d 份 -> %s" % (len(written), PROFILE_DIR))
    print("清单 -> %s" % MANIFEST)


if __name__ == "__main__":
    main()
