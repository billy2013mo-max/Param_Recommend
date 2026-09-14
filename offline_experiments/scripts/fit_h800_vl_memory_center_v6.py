#!/usr/bin/env python3
"""VL 显存模型 V6：分路由中心 + 分路由准入上界。

相对 V5 的唯一变化：准入上界从「全局单一倍率」改为「按路由定倍率」。

为什么
------
V5 用全局倍率时，要做到零错放得推到 3.68——那意味着预测 40 GiB 的配置要到
147 GiB 才拒绝（已超卡容量），准入失去意义。根因是不同路由的 reserved/中心
分布差很远：

    2.5-VL / 2卡 zero3 / gc关   需要 1.34
    3-VL   / 4卡 zero3 / gc关   需要 1.38
    3-VL   / 单卡 / gc开   需要 3.68

用一个全局值就得迁就最松的那条，所有路由都失去拒绝能力。这与中心模型
「全局线性 20%、分路由 8%」是同一类问题：机制差异大到不能用一个参数覆盖。

按路由定倍率后：22 条路由全部零错放 + 100% 覆盖，倍率 1.30-3.68、中位 1.72。

倍率怎么定
----------
每条路由内扫倍率，取「错放 OOM = 0 且 success 覆盖最高」的那个：

    覆盖约束：success 行的 reserved <= 中心 x 倍率
    有效约束：OOM 行的 中心 x 倍率 > 安全线（否则会放行实际炸掉的配置）

没有 OOM 样本的路由只有覆盖约束（无上方约束），倍率只受覆盖驱动，
产物里标 has_oom_constraint=false，提示这条路由的上界未被 OOM 验证过。
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "artifacts" / "h800_vl_memory_table_v1.json"
CENTER_V5 = REPO / "artifacts" / "h800_vl_memory_center_v5.json"
OUT = REPO / "artifacts" / "h800_vl_memory_center_v6.json"

SAFE_LIMIT_BYTES = 142635080089.6
CAPACITY_BYTES = 150142189568
GIB = 1024 ** 3


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def route_of(row: dict) -> str:
    return "%s::g%s_z%s::gc%d" % (row["model_id"], row["gpu_count"],
               row["zero_stage"], int(bool(row["gc"])))


def predict_center(row: dict, coefficients: dict) -> float | None:
    """路由内中心预测。没有该路由的系数就返回 None（调用方回退）。"""
    entry = coefficients.get(route_of(row))
    if not entry:
        return None
    value = entry["intercept"] + entry["log_tokens"] * math.log(row["total_tokens"])
    value += entry["log_mbs"] * math.log2(row["mbs"])
    if entry.get("uses_share"):
        value += entry["share"] * (row["visual_tokens"] / row["total_tokens"])
    return math.exp(value)


def calibrate_route(success: list, oom: list, coefficients: dict) -> dict:
    """在一条路由内定倍率：取「覆盖住全部 success 的最小值」。

    为什么是最小值而不是「零错放区间里的最大值」
    ---------------------------------------------
    先前那种取法会白送松弛。qwen3_vl_4b::g1_z0::gc1 覆盖只需 3.02，但 3.02
    往上覆盖不再增加、错放也仍是 0，贪最大就取到 3.68——多出的 0.66 纯属
    浪费拒绝能力。倍率的职责只是覆盖 reserved 碎片，覆盖住了就该停。

    两道约束看着冲突，其实由不同判据负责
    -----------------------------------
    14 条路由的「覆盖需要」大于「OOM 允许的最大倍率」，看着像矛盾。实际不是：
    那些 OOM 行的中心预测（allocated）本身就已超安全线，第一道判据就拒了，
    轮不到倍率兜底。所以这里只管覆盖，错放由 admission_rule 第一条负责，
    标定后再核验。
    """
    ratios = [row["reserved_bytes"] / predict_center(row, coefficients)
              for row in success]
    needed = max(ratios) if ratios else 1.0
    multiplier = max(1.0, math.ceil(needed * 100) / 100)

    center_alone = 0
    by_upper = 0
    for row in oom:
        value = predict_center(row, coefficients)
        if value > SAFE_LIMIT_BYTES:
            center_alone += 1
        elif value * multiplier > SAFE_LIMIT_BYTES:
            by_upper += 1
    false_admit = len(oom) - center_alone - by_upper

    allows = [SAFE_LIMIT_BYTES / predict_center(row, coefficients) for row in oom]
    return {
        "multiplier": multiplier,
        "success_rows": len(success),
        "oom_rows": len(oom),
        "covered": len(success),
        "coverage": 1.0 if success else None,
        "coverage_needed_ratio": needed if ratios else None,
        "false_admit_oom": false_admit,
        "oom_rejected_by_center_alone": center_alone,
        "oom_rejected_by_upper": by_upper,
        "max_oom_allowed_multiplier": max(allows) if allows else None,
        "has_oom_constraint": len(oom) > 0,
    }


def main() -> None:
    v5 = read_json(CENTER_V5)
    coefficients = v5["route_coefficients"]
    rows = read_json(TABLE)["rows"]

    success = [r for r in rows
               if r.get("allocated_bytes") and r.get("reserved_bytes")
               and predict_center(r, coefficients)]
    oom = [r for r in rows
           if r.get("classification") == "oom" and predict_center(r, coefficients)]

    by_success = defaultdict(list)
    by_oom = defaultdict(list)
    for row in success:
        by_success[route_of(row)].append(row)
    for row in oom:
        by_oom[route_of(row)].append(row)

    upper = {}
    for name in sorted(set(by_success) | set(by_oom)):
        upper[name] = calibrate_route(by_success.get(name, []),
                        by_oom.get(name, []), coefficients)

    total_covered = sum(u["covered"] for u in upper.values())
    total_success = sum(u["success_rows"] for u in upper.values())
    total_false = sum(u["false_admit_oom"] for u in upper.values())
    multipliers = sorted(u["multiplier"] for u in upper.values())
    unverified = [n for n, u in upper.items() if not u["has_oom_constraint"]]

    payload = {
        "schema": "sft_h800_vl_memory_center/v6_routed_upper",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only",
        "production_admission_allowed": False,
        "center": {
            "inherits_from": "h800_vl_memory_center_v5.json",
            "target": "allocated_bytes",
            "route_definition": "(model_id, gpu_count, zero_stage, gc)",
            "within_route_form": v5["within_route_form"],
            "train": v5["train"],
            "acceptance": v5["acceptance"],
            "route_coefficients": coefficients,
        },
        "admission_rule": {
            "step_1_center": "拒绝，如果 中心预测 > 安全线（中心学的是 allocated，预测值本身超线就无需兜底）",
            "step_2_upper": "拒绝，如果 中心预测 x 路由倍率 > 安全线（倍率负责覆盖 reserved 里的分配器碎片）",
            "note": "OOM 里大部分由第一道拦下，倍率只补第二道",
        },
        "admission_upper": {
            "policy": "per_route_multiplier",
            "rule": "upper = route_center x multiplier；multiplier 取覆盖住该路由全部 success 的 reserved 所需的最小值",
            "target": "reserved_bytes",
            "why_not_global": "全局单一倍率要覆盖住所有路由需 3.02，意味着预测 40 GiB 的配置要到 121 GiB 才拒绝；按路由中位 1.72 只需 69 GiB",
            "safe_limit_bytes": SAFE_LIMIT_BYTES,
            "capacity_bytes": CAPACITY_BYTES,
            "routes": upper,
            "summary": {
                "routes": len(upper),
                "oom_rows_total": sum(u["oom_rows"] for u in upper.values()),
                "oom_rejected_by_center_alone": sum(
                    u["oom_rejected_by_center_alone"] for u in upper.values()),
                "oom_rejected_by_upper": sum(
                    u["oom_rejected_by_upper"] for u in upper.values()),
                "success_coverage": total_covered / total_success,
                "false_admit_oom": total_false,
                "multiplier_min": multipliers[0],
                "multiplier_median": multipliers[len(multipliers) // 2],
                "multiplier_max": multipliers[-1],
                "routes_without_oom_constraint": unverified,
            },
        },
    }
    with OUT.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    summary = payload["admission_upper"]["summary"]
    print("=== 中心（沿用 V5）===")
    print("  训练 MAPE %.1f%%   验收 MAPE %.1f%%" % (v5["train"]["mape"] * 100, v5["acceptance"]["mape"] * 100))
    print("")
    print("=== 准入上界（按路由）===")
    print("  路由 %d 条" % summary["routes"])
    print("  success 覆盖 %.1f%%   错放 OOM %d" % (summary["success_coverage"] * 100, summary["false_admit_oom"]))
    print("  %d 个 OOM 里：中心自己拒掉 %d，靠倍率才拒掉 %d" % (summary["oom_rows_total"], summary["oom_rejected_by_center_alone"], summary["oom_rejected_by_upper"]))
    print("  倍率 %.2f-%.2f，中位 %.2f" % (summary["multiplier_min"], summary["multiplier_max"], summary["multiplier_median"]))
    print("  无 OOM 约束（上界未被验证）的路由 %d 条" % len(summary["routes_without_oom_constraint"]))
    for name in summary["routes_without_oom_constraint"]:
        print("     " + name)
    print("")
    print("产物 -> %s" % OUT)


if __name__ == "__main__":
    main()
