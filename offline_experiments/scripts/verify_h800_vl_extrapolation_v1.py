#!/usr/bin/env python3
"""跨模型外推验证：检验「同一条线内结构相同、只有参数量不同 -> 系数可外推」。

这个脚本回答的问题
------------------
给一个没测过的模型，能不能只靠它的参数量/几何量，算出显存与吞吐的系数？
能，就不必每加一个模型都重测一整批；不能，就只能逐模型补数据。

为什么此前的数字不算数
----------------------
2026-09-09 报过跨模型外推 MAPE 54.9%（吞吐）/ 64.7%（显存），那两个数字
测的都不是这个假设：

  一、它们用的是「全局系数」——所有模型混一套参数，参数量只作为一个对数项与
      其它效应线性叠加，不是「先拟每个模型的系数、再拟系数与参数量的关系」。
  二、更要命的是没有杠杆：能单独拟出系数的模型，VL 线只有 3.75B 和 4.44B
      两个（参数量只差 18%），Hybrid 线一个都没有。拿两个几乎重合的点定斜率
      再外推到 8B/32B，误差必然爆掉——那是同义反复，不是检验。

外推验证批（74 格）把 7 个模型跑到同一条配置上，参数量跨度 VL 线 8.9 倍、
Hybrid 线 32 倍，才第一次具备检验条件。

两层做法
--------
第一层  每个 (模型, 路由) 单独拟合，得到该模型自己的系数
          显存  reserved = P + A x mbs x (tokens/4096)^delta
          吞吐  log(tok/s) = a + b log(tokens) + c log2(mbs) + d visual_share
第二层  在同一条路由上，看系数与参数量/几何量的关系有多紧，并做留一模型验证：
        留出一个模型，用同路由其他模型的关系预测它的系数，再算预测误差

留一验证是关键——只看第二层拟合的 R² 会自欺，因为点少时总能拟得很好。

口径
----
权重字节用实测值而非「参数量 x2」：qwen2p5_vl_32b 视觉塔是 FP32，x2 会低估
1.28 GiB，直接污染 P 与权重的关系。
VL 线与 Hybrid 线分开统计，不假设两线共用系数（注意力机制不同）。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "artifacts" / "h800_vl_memory_table_v1.json"
INVENTORY = REPO / "artifacts" / "h800_vl_extrap_probe_model_inventory_v1.json"
OUT = REPO / "artifacts" / "h800_vl_extrapolation_verdict_v1.json"

GIB = 1024.0 ** 3
REF_TOKENS = 4096.0
DELTA = 1.0768          # 全局长度指数，已被 600+ 行钉死
MIN_POINTS = 3          # 拟 P 与 A 至少要 3 个点
MIN_X_SPAN = 2.0        # 自变量跨度太小则 P/A 不可分
MIN_MODELS_PER_ROUTE = 3  # 第二层至少要 3 个模型才谈得上留一
# 物理门禁：常数项含权重本身，小于分片权重即不可能。与准入模型 V2 同一口径
# （留 10% 余量给测量噪声与权重口径差异）。
#
# 不加这道会出事：qwen2p5_vl_7b::g2_z2::gc0 拟出 P = -0.06 GiB（权重 15.45 GiB），
# 那条路由上 mbs 1->2 的 reserved 增量大于 mbs=1 时的全部占用，「微批翻倍则激活
# 翻倍」这条硬约束在那里不成立，形式不适用。留着它，第二层的 P 误差会算出
# -8781%（分母为负），一条坏数据污染整条结论。
MIN_P_OVER_SHARD = 0.9


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def line_of(model_id: str) -> str:
    return "Hybrid" if model_id.startswith("qwen3p5_") else "VL"


def route_of(row: dict) -> str:
    return "g%s_z%s::gc%d" % (row["gpu_count"], row["zero_stage"],
                              int(bool(row["gc"])))


def weight_bytes(model_id: str, entry: dict | None, total_parameters) -> float:
    """优先用实测字节；没有则退回参数量 x2（BF16）。"""
    if entry and entry.get("weight_bytes_measured"):
        return float(entry["weight_bytes_measured"])
    return float(total_parameters) * 2.0


def fit_first_layer(rows: list[dict]) -> dict | None:
    """单个 (模型,路由) 的显存系数。同配置重复测量先去重。"""
    points = {}
    for row in rows:
        x = row["mbs"] * (row["total_tokens"] / REF_TOKENS) ** DELTA
        points[round(x, 6)] = row["reserved_bytes"]
    if len(points) < MIN_POINTS:
        return None
    xs = np.array(sorted(points), dtype=float)
    ys = np.array([points[x] for x in sorted(points)], dtype=float)
    if xs[-1] / xs[0] < MIN_X_SPAN:
        return None
    matrix = np.vstack([np.ones_like(xs), xs]).T
    solution, *_ = np.linalg.lstsq(matrix, ys, rcond=None)
    predicted = matrix @ solution
    ss_res = float(np.sum((ys - predicted) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return {
        "P": float(solution[0]), "A": float(solution[1]),
        "points": len(points), "x_span": float(xs[-1] / xs[0]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
    }


def main() -> None:
    table = read_json(TABLE)
    inventory = {m["id"]: m for m in read_json(INVENTORY)["models"]}
    rows = [r for r in table["rows"]
            if r.get("reserved_bytes") and r.get("total_tokens")]

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], route_of(row))].append(row)

    # ---- 第一层 ----
    first = {}
    for (model_id, route), members in grouped.items():
        fit = fit_first_layer(members)
        if not fit:
            continue
        sample = members[0]
        entry = inventory.get(model_id)
        gpu_count = int(sample["gpu_count"])
        zero = int(sample["zero_stage"])
        total_bytes = weight_bytes(model_id, entry, sample["total_parameters"])
        fit.update({
            "model_id": model_id, "route": route, "line": line_of(model_id),
            "total_parameters": sample["total_parameters"],
            "weight_bytes": total_bytes,
            "shard_bytes": total_bytes / (gpu_count if zero == 3 else 1),
            "layers": sample.get("num_hidden_layers"),
            "hidden": sample.get("hidden_size"),
            "vision_tower_parameters": sample.get("vision_tower_parameters"),
            "weight_bytes_source": ("measured" if entry
                                    and entry.get("weight_bytes_measured")
                                    else "parameters_x2"),
        })
        first[(model_id, route)] = fit

    # 物理门禁：剔除常数项装不下分片权重的组。这类组不是「样本少」，而是加法
    # 形式在那条路由上不适用（反推可证常数项为负），留着会污染第二层统计。
    rejected = {key: f for key, f in first.items()
                if f["P"] / f["shard_bytes"] < MIN_P_OVER_SHARD}
    for key in rejected:
        del first[key]

    print("=== 第一层：能单独拟出系数的 (模型,路由) 共 %d 组 ===" % len(first))
    if rejected:
        print("   物理门禁剔除 %d 组（常数项 < 分片权重 x %.1f，形式不适用）：" % (
            len(rejected), MIN_P_OVER_SHARD))
        for key, f in sorted(rejected.items()):
            print("      %-18s %-12s P %6.2fG  分片权重 %6.2fG  比值 %.3f" % (
                f["model_id"], f["route"], f["P"] / GIB,
                f["shard_bytes"] / GIB, f["P"] / f["shard_bytes"]))
    print("%-18s %-12s %5s %7s %6s %9s %9s %8s" % (
        "模型", "路由", "点数", "跨度", "R²", "P(GiB)", "A(GiB)", "P/权重"))
    for key in sorted(first, key=lambda k: (first[k]["line"],
                                            first[k]["total_parameters"])):
        f = first[key]
        print("%-18s %-12s %5d %6.1fx %6.3f %9.2f %9.2f %8.3f" % (
            f["model_id"], f["route"], f["points"], f["x_span"],
            f["r2"] if f["r2"] is not None else float("nan"),
            f["P"] / GIB, f["A"] / GIB, f["P"] / f["shard_bytes"]))

    # ---- 第二层：留一模型外推 ----
    by_route_line = defaultdict(list)
    for f in first.values():
        by_route_line[(f["line"], f["route"])].append(f)

    verdicts = []
    for (line, route), members in sorted(by_route_line.items()):
        if len(members) < MIN_MODELS_PER_ROUTE:
            verdicts.append({"line": line, "route": route,
                             "models": len(members), "verdict": "样本不足，无法检验"})
            continue
        errors_p, errors_a = [], []
        detail = []
        for held in members:
            others = [m for m in members if m["model_id"] != held["model_id"]]
            # P 用「常数项 / 分片权重」的中位比值外推
            ratios = [m["P"] / m["shard_bytes"] for m in others]
            p_pred = float(np.median(ratios)) * held["shard_bytes"]
            # A 用「激活系数 / (层数 x 隐藏维度)」的中位比值外推
            geoms = [m["A"] / (m["layers"] * m["hidden"]) for m in others
                     if m.get("layers") and m.get("hidden")]
            a_pred = (float(np.median(geoms)) * held["layers"] * held["hidden"]
                      if geoms and held.get("layers") and held.get("hidden")
                      else None)
            err_p = abs(p_pred - held["P"]) / held["P"]
            errors_p.append(err_p)
            err_a = None
            if a_pred is not None and held["A"] > 0:
                err_a = abs(a_pred - held["A"]) / held["A"]
                errors_a.append(err_a)
            detail.append({
                "held_out": held["model_id"],
                "params_b": held["total_parameters"] / 1e9,
                "P_actual_gib": held["P"] / GIB, "P_pred_gib": p_pred / GIB,
                "P_error": err_p,
                "A_actual_gib": held["A"] / GIB,
                "A_pred_gib": a_pred / GIB if a_pred is not None else None,
                "A_error": err_a,
            })
        verdicts.append({
            "line": line, "route": route, "models": len(members),
            "P_mape": float(np.mean(errors_p)),
            "A_mape": float(np.mean(errors_a)) if errors_a else None,
            "P_worst": float(np.max(errors_p)),
            "param_span": (max(m["total_parameters"] for m in members)
                           / min(m["total_parameters"] for m in members)),
            "detail": detail,
            "verdict": None,
        })

    print("")
    print("=== 第二层：留一模型外推（预测被留出模型的系数）===")
    print("%-8s %-12s %6s %8s %9s %9s" % (
        "线", "路由", "模型数", "参数跨度", "P误差", "A误差"))
    for v in verdicts:
        if v.get("verdict") == "样本不足，无法检验":
            print("%-8s %-12s %6d   %s" % (v["line"], v["route"], v["models"],
                                           v["verdict"]))
            continue
        print("%-8s %-12s %6d %7.1fx %8.1f%% %8s" % (
            v["line"], v["route"], v["models"], v["param_span"],
            100 * v["P_mape"],
            "%.1f%%" % (100 * v["A_mape"]) if v["A_mape"] is not None else "-"))

    # ---- 结论 ----
    testable = [v for v in verdicts if v.get("P_mape") is not None]
    payload = {
        "schema": "sft_h800_vl_extrapolation_verdict/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "同一条线内，路由系数能否用参数量/几何量外推到没测过的模型",
        "method": {
            "layer_1": "每个 (模型,路由) 单独拟 P 与 A",
            "layer_2": "同路由内留一模型，用其他模型的比值中位数预测被留出者",
            "why_leave_one_out": "只看第二层拟合优度会自欺——点少时总能拟得很好",
        },
        "caveats": {
            "weight_bytes": "用实测字节；qwen2p5_vl_32b 视觉塔 FP32，x2 会低估 1.28 GiB",
            "lines_separate": "VL 与 Hybrid 分开，不假设共用系数（注意力机制不同）",
            "moe_excluded": "本批无 MoE；已知 MoE 在全局系数下误差 294%，需单独形式",
            "delta_fixed": "长度指数取全局 %.4f，未逐模型重估" % DELTA,
            "physical_gate": "剔除常数项 < 分片权重 x %.1f 的组（形式不适用，"
                             "反推可证常数项为负）" % MIN_P_OVER_SHARD,
        },
        "physically_rejected": [
            {"model_id": f["model_id"], "route": f["route"],
             "P_gib": f["P"] / GIB, "shard_bytes_gib": f["shard_bytes"] / GIB,
             "ratio": f["P"] / f["shard_bytes"]}
            for f in rejected.values()],
        "first_layer": [
            {k: v for k, v in f.items()} for f in first.values()],
        "second_layer": verdicts,
        "summary": {
            "routes_testable": len(testable),
            "routes_insufficient": len(verdicts) - len(testable),
            "P_mape_overall": (float(np.mean([v["P_mape"] for v in testable]))
                               if testable else None),
            "A_mape_overall": (float(np.mean([v["A_mape"] for v in testable
                                              if v["A_mape"] is not None]))
                               if any(v["A_mape"] is not None for v in testable)
                               else None),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    print("")
    if testable:
        p_all = payload["summary"]["P_mape_overall"]
        a_all = payload["summary"]["A_mape_overall"]
        print("可检验的路由 %d 条：常数项外推 MAPE %.1f%%，激活系数 %s" % (
            len(testable), 100 * p_all,
            "%.1f%%" % (100 * a_all) if a_all is not None else "不可用"))
        print("")
        print("判读参考：常数项误差 <15%% 说明它确实由权重决定、可外推；")
        print("激活系数误差远大于常数项属正常——它含视觉塔与实现细节，")
        print("不是纯几何量。两者都要看留一验证，不看第二层拟合优度。")
    else:
        print("没有任何路由达到可检验条件（同路由需 >=%d 个模型）"
              % MIN_MODELS_PER_ROUTE)
    print("")
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
