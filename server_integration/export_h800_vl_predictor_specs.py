#!/usr/bin/env python3
"""Export the H800 VL predictor spec for a Go reimplementation.

产出了三头（memory_center, throughput, admission）的系数、公式和特征定义，
供平台侧用 Go 重写。配套 test_vectors 给出每条请求的完整中间特征向量。

与 RTX4090 的区别：
  - 三层路由回归（route → parent → global），不是扁平线性模型
  - admission 是独立的加法物理形式，不是显存模型的安全边界
  - admission 外推从几何列参数化 P/A，覆盖支持域外的 19 条路由
"""
from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "offline_experiments" / "artifacts"
OUT_DIR = REPO / "server_integration" / "artifacts"
VECTOR_DIR = REPO / "server_integration" / "testdata"

CAPACITY_BYTES = 150142189568.0
SAFE_LIMIT_BYTES = 142635080089.6
GIB = 1024.0 ** 3
REF_TOKENS = 4096.0
Z_P95 = 1.645


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


# -- feature functions (mirror fit scripts exactly) --

def route_of(model_id, gpu_count, zero_stage, gc):
    return f"{model_id}::g{gpu_count}_z{zero_stage}::gc{int(bool(gc))}"


def parent_route_of(model_id, gpu_count, zero_stage):
    return f"{model_id}::g{gpu_count}_z{zero_stage}"


def memory_route_features(row, uses_share):
    base = [1.0, math.log(row["total_tokens"]), math.log2(row["mbs"])]
    if uses_share:
        base.append(row["visual_tokens"] / row["total_tokens"])
    return base


def memory_global_features(row):
    params = row.get("language_parameters") or row["total_parameters"]
    gc = 1.0 if row["gc"] else 0.0
    share = row["visual_tokens"] / row["total_tokens"]
    return [1.0, math.log(params), math.log(row["total_tokens"]),
            math.log2(row["mbs"]), gc, math.log2(row["gpu_count"]),
            1.0 if row["zero_stage"] == 2 else 0.0,
            1.0 if row["zero_stage"] == 3 else 0.0,
            share * gc]


def throughput_route_features(row):
    return [1.0, math.log(row["total_tokens"]), math.log2(row["mbs"]),
            row["visual_tokens"] / row["total_tokens"]]


def throughput_global_features(row):
    params = row.get("language_parameters") or row["total_parameters"]
    return [1.0, math.log(params), math.log(row["total_tokens"]),
            math.log2(row["mbs"]), 1.0 if row["gc"] else 0.0,
            math.log2(row["gpu_count"]),
            1.0 if row["zero_stage"] == 2 else 0.0,
            1.0 if row["zero_stage"] == 3 else 0.0,
            row["visual_tokens"] / row["total_tokens"]]


def admission_features(route_name, geo):
    model_id, topology, gc_str = route_name.split("::")
    gpu_count = int(topology.split("_")[0][1:])
    zero_stage = int(topology.split("_")[1][1:])
    gc = 1.0 if gc_str == "gc1" else 0.0
    params = geo.get("language_parameters") or geo.get("total_parameters")
    return [1.0, math.log(params), gc, math.log2(gpu_count),
            1.0 if zero_stage == 2 else 0.0, 1.0 if zero_stage == 3 else 0.0]


def predict_memory(row, mem_art):
    rc = mem_art["route_coefficients"]
    pc = mem_art["parent_coefficients"]
    gm = mem_art["global_model"]
    name = route_of(row["model_id"], row["gpu_count"], row["zero_stage"], row["gc"])
    if name in rc:
        entry = rc[name]
        feat = memory_route_features(row, entry["uses_share"])
        coeffs = [entry["intercept"], entry["log_tokens"], entry["log_mbs"]]
        if entry["uses_share"]:
            coeffs.append(entry["share"])
        level = "route"
    else:
        parent = parent_route_of(row["model_id"], row["gpu_count"], row["zero_stage"])
        if parent in pc:
            entry = pc[parent]
            feat = memory_route_features(row, False)
            coeffs = [entry["intercept"], entry["log_tokens"], entry["log_mbs"]]
            level = "parent"
        else:
            feat = memory_global_features(row)
            coeffs = gm["coefficients"]
            level = "global"
    log_pred = float(np.dot(feat, coeffs))
    return math.exp(log_pred), level, feat


def predict_throughput(row, tp_art):
    rc = tp_art["route_coefficients"]
    pc = tp_art["parent_coefficients"]
    gm = tp_art["global_model"]
    name = route_of(row["model_id"], row["gpu_count"], row["zero_stage"], row["gc"])
    if name in rc:
        entry = rc[name]
        feat = throughput_route_features(row)
        coeffs = [entry["intercept"], entry["log_tokens"], entry["log_mbs"], entry["share"]]
        level = "route"
    else:
        parent = parent_route_of(row["model_id"], row["gpu_count"], row["zero_stage"])
        if parent in pc:
            entry = pc[parent]
            feat = throughput_route_features(row)
            coeffs = [entry["intercept"], entry["log_tokens"], entry["log_mbs"], entry["share"]]
            level = "parent"
        else:
            feat = throughput_global_features(row)
            coeffs = gm["coefficients"]
            level = "global"
    log_pred = float(np.dot(feat, coeffs))
    return math.exp(log_pred), level, feat


def predict_admission(row, adm_art, ext_art):
    name = route_of(row["model_id"], row["gpu_count"], row["zero_stage"], row["gc"])
    rp = adm_art["route_parameters"]
    ext = ext_art["extrapolated_routes"]
    withheld = ext_art.get("withheld_routes", {})
    delta = adm_art["delta"]
    sigma = adm_art["sigma"]

    if name in rp:
        P = rp[name]["P_bytes"]
        A = rp[name]["A_bytes"]
        source = "route"
    elif name in ext:
        P = ext[name]["P_bytes"]
        A = ext[name]["A_bytes"]
        source = "extrapolated"
    else:
        return None, "withheld", None, None, None

    pred = P + A * row["mbs"] * (row["total_tokens"] / REF_TOKENS) ** delta
    upper = pred * math.exp(Z_P95 * sigma)
    admitted = upper <= SAFE_LIMIT_BYTES
    return pred, source, P, A, upper


def build_spec(mem_art, tp_art, adm_art, ext_art):
    return {
        "line": "h800_vl",
        "status": "candidate_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "供平台侧用 Go 重写：本文件给出三头（memory_center, throughput, "
            "admission）的公式、常量、系数与特征定义；配套 test_vectors "
            "给出每条请求的完整中间特征向量，用于逐位对齐。"
        ),
        "card_memory_bytes": CAPACITY_BYTES,
        "safe_limit_bytes": SAFE_LIMIT_BYTES,
        "heads": {
            "memory_center": {
                "target": "allocated_bytes",
                "formula": {
                    "route_gc_off": "log(alloc) = a + b*log(tokens) + c*log2(mbs)",
                    "route_gc_on": "log(alloc) = a + b*log(tokens) + c*log2(mbs) + d*share",
                    "parent": "log(alloc) = a + b*log(tokens) + c*log2(mbs)  # no share",
                    "global": "log(alloc) = dot(global_features, global_coefficients)",
                    "output": "exp(log_pred)  # bytes",
                },
                "route_definition": "(model_id, gpu_count, zero_stage, gc)",
                "fallback_policy": "route -> parent -> global",
                "route_coefficients": mem_art["route_coefficients"],
                "parent_coefficients": mem_art["parent_coefficients"],
                "global_model": mem_art["global_model"],
                "quality": mem_art["quality"],
                "train_by_fit_level": mem_art["train"]["by_fit_level"],
            },
            "throughput": {
                "target": "effective_tokens_per_second (single rank)",
                "target_note": "多卡总吞吐 = 单 rank × 卡数",
                "formula": {
                    "route": "log(tps) = a + b*log(tokens) + c*log2(mbs) + d*share",
                    "parent": "log(tps) = a + b*log(tokens) + c*log2(mbs) + d*share",
                    "global": "log(tps) = dot(global_features, global_coefficients)",
                    "output": "exp(log_pred)  # tokens/s",
                },
                "route_definition": "(model_id, gpu_count, zero_stage, gc)",
                "fallback_policy": "route -> parent -> global",
                "route_coefficients": tp_art["route_coefficients"],
                "parent_coefficients": tp_art["parent_coefficients"],
                "global_model": tp_art["global_model"],
                "quality": tp_art["quality"],
                "train_by_fit_level": tp_art["train"]["by_fit_level"],
            },
            "admission": {
                "target": "装不装得下（reserved 是否超安全线）",
                "form": "reserved = P + A * mbs * (tokens/4096)^delta",
                "delta": adm_art["delta"],
                "sigma": adm_art["sigma"],
                "admission_rule": f"reject if pred * exp({Z_P95} * sigma) > safe_limit",
                "safe_limit_bytes": SAFE_LIMIT_BYTES,
                "route_parameters": adm_art["route_parameters"],
                "extrapolation": {
                    "method": ext_art["method"],
                    "P_model": ext_art["P_model"],
                    "A_model": ext_art["A_model"],
                    "extrapolated_routes": ext_art["extrapolated_routes"],
                    "withheld_routes": ext_art["withheld_routes"],
                    "false_admit_oom_check": ext_art["false_admit_oom_check"],
                    "extrapolation_boundary": ext_art["extrapolation_boundary"],
                },
            },
        },
        "quality_summary": {
            "memory": {
                "train_mape": mem_art["train"]["mape"],
                "acceptance_mape": mem_art["acceptance"]["mape"],
                "global_train_mape": mem_art["train"]["by_fit_level"].get("global", {}).get("mape"),
            },
            "throughput": {
                "train_mape": tp_art["train"]["mape"],
                "acceptance_mape": tp_art["acceptance_source_isolated"]["mape"],
                "global_train_mape": tp_art["train"]["by_fit_level"].get("global", {}).get("mape"),
            },
            "admission": {
                "false_admit_oom_in_domain": ext_art["false_admit_oom_check"]["in_domain"]["false_admit"],
                "false_admit_oom_out_of_domain": ext_art["false_admit_oom_check"]["out_of_domain"]["false_admit_after_withhold"],
                "routes_extrapolated": ext_art["extrapolation_boundary"]["routes_extrapolated"],
                "routes_withheld": ext_art["extrapolation_boundary"]["routes_withheld"],
            },
        },
        "limitations": [
            "显存全局层 train MAPE 29%，吞吐全局层 47%——route/parent 层覆盖时精度好（route 7-10%），"
            "但外推到完全没见过的 (model, gpu, zero, gc) 组合时全局层误差大。"
            "Go 侧应优先用 route/parent 层，全局层只在兜底时使用。",
            "admission 外推用了安全包络（max positive residual），P 膨胀 ~1.6×、A 膨胀 ~5.1×。"
            "保守（多拒），但 false_admit_oom = 0 已验证。",
            "只覆盖 H800 80G，12 个 VL 模型。qwen2p5_vl_7b::g2_z2::gc0 被物理门禁拒绝"
            "（加法形式不适用），返回 None。",
        ],
    }


def build_test_vectors(mem_art, tp_art, adm_art, ext_art, geo_by_model):
    """生成测试向量，覆盖 in-domain / out-of-domain / extrapolated 路由。"""
    # 选代表性的模型/配置组合
    models = ["qwen2p5_vl_3b", "qwen3p5_9b", "qwen3p5_0p8b", "qwen2p5_vl_32b"]
    gpu_counts = [1, 2, 4]
    zero_stages = [0, 2, 3]
    gcs = [0, 1]
    mbs_values = [1, 4]
    token_values = [4096, 32768]

    vectors = []
    for model_id, gpu_count, zero_stage, gc, mbs, tokens in itertools.product(
            models, gpu_counts, zero_stages, gcs, mbs_values, token_values):
        geo = geo_by_model.get(model_id, {})
        params = geo.get("language_parameters") or geo.get("total_parameters")
        if not params:
            continue
        # 构造一个 row dict（模拟建模表的行）
        visual_tokens = int(params * 0.1)  # 10% visual
        total_tokens = tokens
        if visual_tokens > total_tokens:
            visual_tokens = total_tokens // 4
        row = {
            "model_id": model_id,
            "gpu_count": gpu_count,
            "zero_stage": zero_stage,
            "gc": gc,
            "mbs": mbs,
            "total_tokens": total_tokens,
            "visual_tokens": visual_tokens,
            "language_parameters": geo.get("language_parameters"),
            "total_parameters": geo.get("total_parameters"),
        }
        name = route_of(model_id, gpu_count, zero_stage, gc)

        # Memory
        mem_pred, mem_level, mem_feat = predict_memory(row, mem_art)
        # Throughput
        tp_pred, tp_level, tp_feat = predict_throughput(row, tp_art)
        # Admission
        adm_pred, adm_source, P, A, upper = predict_admission(row, adm_art, ext_art)

        entry = {
            "request": {
                "model_id": model_id,
                "gpu_count": gpu_count,
                "zero_stage": zero_stage,
                "gc": gc,
                "mbs": mbs,
                "total_tokens": total_tokens,
                "visual_tokens": visual_tokens,
            },
            "route": name,
            "result": {
                "memory_center_bytes": mem_pred,
                "memory_center_level": mem_level,
                "throughput_tps": tp_pred,
                "throughput_level": tp_level,
                "admission_pred_bytes": adm_pred,
                "admission_source": adm_source,
                "admission_upper_bytes": upper,
                "admitted": (upper <= SAFE_LIMIT_BYTES) if upper is not None else None,
            },
            "intermediates": {
                "memory_feature_vector": mem_feat,
                "throughput_feature_vector": tp_feat,
            },
        }
        if P is not None:
            entry["intermediates"]["admission_P_bytes"] = P
            entry["intermediates"]["admission_A_bytes"] = A
        vectors.append(entry)
    return vectors


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    mem_art = read_json(ART / "h800_vl_memory_center_v5.json")
    tp_art = read_json(ART / "h800_vl_throughput_center_v1.json")
    adm_art = read_json(ART / "h800_vl_admission_v2.json")
    ext_art = read_json(ART / "h800_vl_admission_extrapolation_v1.json")

    # geometry lookup
    table = read_json(ART / "h800_vl_memory_table_v1.json")
    geo_by_model: dict[str, dict] = {}
    for row in table["rows"]:
        mid = row["model_id"]
        if mid not in geo_by_model:
            geo_by_model[mid] = {
                "language_parameters": row.get("language_parameters"),
                "total_parameters": row.get("total_parameters"),
            }

    spec = build_spec(mem_art, tp_art, adm_art, ext_art)
    spec_path = OUT_DIR / "h800_vl_predictor_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    vectors = build_test_vectors(mem_art, tp_art, adm_art, ext_art, geo_by_model)
    vec_path = VECTOR_DIR / "test_vectors_h800_vl.json"
    vec_path.write_text(json.dumps({
        "schema": "sft_h800_vl_go_port_vectors/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": str(spec_path.relative_to(REPO)),
        "how_to_use": (
            "对每条 request 用 Go 实现算出 memory/throughput 特征向量和 admission 预测，"
            "先跟 intermediates 里的向量对齐；对齐之后再比 result。"
            "先比特征能直接定位到是哪一项算错了。"
        ),
        "vectors": vectors,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary
    print(json.dumps({
        "spec": str(spec_path.relative_to(REPO)),
        "vectors_file": str(vec_path.relative_to(REPO)),
        "vectors": len(vectors),
        "memory_routes": len(mem_art["route_coefficients"]),
        "memory_parents": len(mem_art["parent_coefficients"]),
        "throughput_routes": len(tp_art["route_coefficients"]),
        "throughput_parents": len(tp_art["parent_coefficients"]),
        "admission_routes": len(adm_art["route_parameters"]),
        "extrapolated_routes": len(ext_art["extrapolated_routes"]),
        "withheld_routes": len(ext_art["withheld_routes"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
