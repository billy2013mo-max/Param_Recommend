#!/usr/bin/env python3
"""外推验证批的增量模型清单：在既有清单上补 qwen2p5_vl_32b 一条。

为什么要补
----------
显存表里每行的几何字段（层数、隐藏维度、总参数量、视觉塔参数量）全部来自
模型清单，按 model_id 索引。qwen2p5_vl_32b 从没测过、不在清单里，它的表行
几何字段会全空——准入模型算权重没依据，外推验证也用不了。

为什么写成 sidecar 而不是改原文件
---------------------------------
原清单的 sha256 被已冻结的队列绑定。原文件只读、不覆盖，新队列引用新文件。
做法与 build_h800_v3_inventory_with_hybrid_v1 一致。

数据来源
--------
几何      config.json
参数量    safetensors 头部 shape 求和（不 import torch、不加载模型）
分量      按 tensor key 前缀归类（visual.merger -> 投影器，visual -> 视觉塔，其余 -> 语言侧）
dtype     safetensors 头部实测。qwen2p5_vl_32b 视觉塔是 FP32、语言侧 BF16，
          所以 checkpoint_bytes 按实测算，不能用「参数量 x 2」
tokenizer 实际加载 tokenizer 读

只跑 CPU。不建审批，不启动 GPU。
"""
from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from pathlib import Path

from common import ARTIFACT_DIR, read_json, write_json, sha256_json

BASE = ARTIFACT_DIR / "h800_qwen35_vl_supplement_model_inventory_v1.json"
OUT = ARTIFACT_DIR / "h800_vl_extrap_probe_model_inventory_v1.json"

NEW_MODEL = {
    "id": "qwen2p5_vl_32b",
    "nominal_scale_b": 32,
    "path": "/wanqing-models/Qwen2.5-VL-32B-Instruct",
    "family": "qwen2p5_vl",
    "template": "qwen2_vl",
    "train_types": ["lora"],
    "image_min_pixels": 3136,
    "enable_liger_kernel": True,
}

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
               "I64": 8, "I32": 4, "U8": 1, "I8": 1, "BOOL": 1}


def read_headers(root: Path):
    """返回 [(文件名, 头部 dict)]，按文件名排序。"""
    out = []
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        out.append((path, header))
    return out


def classify(key: str) -> str:
    if key.startswith("visual.merger") or key.startswith("model.visual.merger"):
        return "projector_or_merger"
    if key.startswith("visual") or key.startswith("model.visual"):
        return "vision_tower"
    return "language_model_or_other"


def infer_vision_depth(headers) -> int | None:
    """config 没写 depth 时，从 visual.blocks.<N> 的最大索引推断视觉塔层数。

    qwen2p5_vl_32b 的 vision_config 里没有 depth 字段（7B 有），但权重结构里
    有，所以按权重推断而不是留空——显存表的 vision_depth 字段取自这里。
    """
    import re
    pattern = re.compile(r"^(?:model\.)?visual\.blocks\.(\d+)\.")
    largest = -1
    for _, header in headers:
        for key in header:
            if key == "__metadata__":
                continue
            match = pattern.match(key)
            if match:
                largest = max(largest, int(match.group(1)))
    return largest + 1 if largest >= 0 else None


def main() -> None:
    from transformers import AutoTokenizer, AutoConfig

    root = Path(NEW_MODEL["path"])
    headers = read_headers(root)
    if not headers:
        raise FileNotFoundError("找不到 safetensors：%s" % root)

    components = {"language_model_or_other": 0, "vision_tower": 0,
                  "projector_or_merger": 0}
    total = 0
    tensor_count = 0
    checkpoint_bytes = 0
    manifest = []
    for path, header in headers:
        stat = path.stat()
        manifest.append({"name": path.name, "size": stat.st_size,
                         "mtime_ns": stat.st_mtime_ns})
        checkpoint_bytes += stat.st_size
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            count = 1
            for dim in meta.get("shape") or []:
                count *= dim
            total += count
            tensor_count += 1
            components[classify(key)] += count

    config = AutoConfig.from_pretrained(str(root))
    raw_config = json.loads((root / "config.json").read_text())
    vision_raw = raw_config.get("vision_config") or {}
    preprocessor = json.loads((root / "preprocessor_config.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(root))
    vocab_size = raw_config.get("vocab_size")

    entry = dict(NEW_MODEL)
    entry["tokenizer_path"] = NEW_MODEL["path"]
    entry.update({
        "actual_parameters": total,
        "actual_parameters_b": total / 1e9,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_shards": len(manifest),
        "checkpoint_manifest": manifest,
        "model_directory_name": root.name,
        "model_identity": str(root),
        "revision_policy": "The configured local model directory name/path is "
                           "the authoritative model identity.",
        "tokenizer_identity": str(root),
        "tokenizer_directory_name": root.name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "tokenizer_max_id": max(tokenizer.get_vocab().values()),
        "tokenizer_fits_model_embeddings": (
            max(tokenizer.get_vocab().values()) < int(vocab_size)),
        "tensor_count": tensor_count,
        "model_type": raw_config.get("model_type"),
        "architectures": raw_config.get("architectures"),
        "architecture_role": "vision_language",
        "is_vision_language": True,
        "vision_config": vision_raw,
        "vision_geometry": {
            "model_type": vision_raw.get("model_type"),
            "depth": vision_raw.get("depth") or infer_vision_depth(headers),
            "depth_source": ("config" if vision_raw.get("depth")
                             else "inferred_from_visual_blocks_index"),
            "hidden_size": vision_raw.get("hidden_size"),
            "intermediate_size": vision_raw.get("intermediate_size"),
            "num_heads": vision_raw.get("num_heads"),
            "patch_size": vision_raw.get("patch_size") or preprocessor.get("patch_size"),
            "spatial_merge_size": (vision_raw.get("spatial_merge_size")
                                   or preprocessor.get("merge_size")),
            "temporal_patch_size": (vision_raw.get("temporal_patch_size")
                                    or preprocessor.get("temporal_patch_size")),
            "out_hidden_size": vision_raw.get("out_hidden_size"),
            "in_channels": vision_raw.get("in_chans"),
            "num_position_embeddings": vision_raw.get("num_position_embeddings"),
            "fields_absent_in_config": sorted(
                k for k in ("depth", "num_heads", "patch_size", "spatial_merge_size")
                if vision_raw.get(k) is None),
        },
        "component_parameter_estimates": components,
        "vision_parameter_estimate": (components["vision_tower"]
                                      + components["projector_or_merger"]),
        "hidden_size": raw_config.get("hidden_size"),
        "intermediate_size": raw_config.get("intermediate_size"),
        "num_hidden_layers": raw_config.get("num_hidden_layers"),
        "num_attention_heads": raw_config.get("num_attention_heads"),
        "num_key_value_heads": raw_config.get("num_key_value_heads"),
        "head_dim": raw_config.get("head_dim") or (
            raw_config["hidden_size"] // raw_config["num_attention_heads"]),
        "vocab_size": vocab_size,
        "max_position_embeddings": raw_config.get("max_position_embeddings"),
        "torch_dtype": raw_config.get("torch_dtype") or raw_config.get("dtype"),
        "geometry_source": "top_level",
    })

    # dtype 构成与实测权重字节（视觉塔 FP32，不能用 参数量 x2）
    dtypes: dict[str, int] = {}
    weight_bytes = 0
    for _, header in headers:
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            count = 1
            for dim in meta.get("shape") or []:
                count *= dim
            code = meta.get("dtype")
            dtypes[code] = dtypes.get(code, 0) + count
            weight_bytes += count * DTYPE_BYTES.get(code, 2)
    entry["dtype_parameter_counts"] = dtypes
    entry["weight_bytes_measured"] = weight_bytes
    entry["weight_bytes_naive_x2"] = total * 2
    entry["weight_bytes_note"] = (
        "视觉塔为 FP32、语言侧 BF16，按「参数量 x2」会低估 %.2f GiB"
        % ((weight_bytes - total * 2) / 1024 ** 3))
    # config 声明的 dtype 与磁盘实测不符，按几何清单的既定原则以磁盘为准
    declared_vision_dtype = vision_raw.get("torch_dtype") or vision_raw.get("dtype")
    if declared_vision_dtype and "F32" in dtypes:
        entry["vision_dtype_conflict"] = {
            "config_declares": declared_vision_dtype,
            "measured_from_safetensors": "F32",
            "resolution": "以磁盘权重为准（与《模型几何参数清单》同一原则）",
        }

    base = read_json(BASE)
    existing = {m["id"] for m in base["models"]}
    if entry["id"] in existing:
        raise ValueError("基础清单里已有 %s，不该走增量" % entry["id"])

    payload = {
        "schema": "sft_h800_vl_extrap_probe_model_inventory/v1",
        "campaign_id": "h800_vl_extrap_probe_20260909_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_training_started": False,
        "base_inventory_path": str(BASE),
        "base_inventory_sha256": sha256_json(base),
        "additive_only": True,
        "added_model_ids": [entry["id"]],
        "why": "qwen2p5_vl_32b 从未测过、不在基础清单里；显存表的几何字段全部"
               "按 model_id 从清单取，缺条目会导致该模型的表行几何为空",
        "fixed_lora": base["fixed_lora"],
        "models": base["models"] + [entry],
    }
    payload["report_sha256"] = sha256_json(payload)
    write_json(OUT, payload)

    print("基础清单 %d 个模型 -> 新清单 %d 个" % (
        len(base["models"]), len(payload["models"])))
    print("")
    print("新增条目 qwen2p5_vl_32b：")
    for key in ("actual_parameters", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size",
                "tensor_count", "checkpoint_shards", "tokenizer_class",
                "tokenizer_size", "tokenizer_fits_model_embeddings"):
        print("   %-32s %s" % (key, entry[key]))
    print("   %-32s %s" % ("component_parameter_estimates",
                           json.dumps(entry["component_parameter_estimates"])))
    print("   %-32s %s" % ("vision_geometry.depth/hidden/patch",
                           [entry["vision_geometry"]["depth"],
                            entry["vision_geometry"]["hidden_size"],
                            entry["vision_geometry"]["patch_size"]]))
    print("   %-32s %d  (naive x2 = %d, 差 %.2f GiB)" % (
        "weight_bytes_measured", entry["weight_bytes_measured"],
        entry["weight_bytes_naive_x2"],
        (entry["weight_bytes_measured"] - entry["weight_bytes_naive_x2"]) / 1024 ** 3))
    print("")
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
