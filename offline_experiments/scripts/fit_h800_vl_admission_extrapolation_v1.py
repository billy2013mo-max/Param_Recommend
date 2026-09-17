#!/usr/bin/env python3
"""VL 准入外推：从几何列参数化 P_bytes/A_bytes，外推到支持域外的路由。

准入 V2 在 51 条路由上服务。剩 20 条返回 None（超出支持域）。
本脚本从 51 条已拟合的 P/A 出发，用几何特征回归 log(P)/log(A)，
再用安全包络（膨胀到覆盖最大正残差）外推到域外路由。

安全约束：false_admit_oom = 0。外推后如果有 OOM 行被放行
（上界 < 安全线），把那条路由从外推域拿掉（拒答优于错放）。

形式
----
    log(P) = w . features(params, zero, gc, gpu_count)
    log(A) = v . features(params, zero, gc, gpu_count)

    P_upper = exp(log(P_pred) + max_positive_residual_P)
    A_upper = exp(log(A_pred) + max_positive_residual_A)

    pred = P_upper + A_upper * mbs * (tokens/4096)^delta
    upper = pred * exp(1.645 * sigma)
    拒绝，如果 upper > 安全线
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "artifacts" / "h800_vl_memory_table_v1.json"
ADMISSION_V2 = REPO / "artifacts" / "h800_vl_admission_v2.json"
OUT = REPO / "artifacts" / "h800_vl_admission_extrapolation_v1.json"

CAPACITY_BYTES = 150142189568.0
SAFE_LIMIT_BYTES = 142635080089.6
GIB = 1024.0 ** 3
REF_TOKENS = 4096.0
Z_P95 = 1.645

FEATURE_ORDER = ["const", "log_params", "gc", "log2_gpu_count",
                 "is_zero2", "is_zero3"]


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def parse_route(name: str) -> tuple[str, int, int, int]:
    model_id, topology, gc_str = name.split("::")
    gpu_count = int(topology.split("_")[0][1:])
    zero_stage = int(topology.split("_")[1][1:])
    gc = int(gc_str == "gc1")
    return model_id, gpu_count, zero_stage, gc


def features(name: str, geo: dict) -> list[float]:
    model_id, gpu_count, zero_stage, gc = parse_route(name)
    params = geo.get("language_parameters") or geo.get("total_parameters")
    return [
        1.0,
        math.log(params),
        float(gc),
        math.log2(gpu_count),
        1.0 if zero_stage == 2 else 0.0,
        1.0 if zero_stage == 3 else 0.0,
    ]


def solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    ridge = 1e-8 * np.eye(matrix.shape[1])
    return np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ target)


def main() -> None:
    adm = read_json(ADMISSION_V2)
    delta = adm["delta"]
    sigma = adm["sigma"]
    route_params = adm["route_parameters"]
    routes_in = adm["support_domain"]["routes_in"]
    routes_out = adm["support_domain"]["routes_out"]
    rejected_by_physical = set(adm["support_domain"]["rejected_by_physical_check"])

    table = read_json(TABLE)
    geo_by_model: dict[str, dict] = {}
    for row in table["rows"]:
        mid = row["model_id"]
        if mid not in geo_by_model:
            geo_by_model[mid] = {
                "language_parameters": row.get("language_parameters"),
                "total_parameters": row.get("total_parameters"),
            }

    # -- 回归 log(P) / log(A) on 51 in-domain routes --
    X_list, logP_list, logA_list = [], [], []
    skipped = []
    for name in routes_in:
        mid = name.split("::")[0]
        geo = geo_by_model.get(mid, {})
        params = geo.get("language_parameters") or geo.get("total_parameters")
        if not params:
            skipped.append(name)
            continue
        X_list.append(features(name, geo))
        logP_list.append(math.log(route_params[name]["P_bytes"]))
        logA_list.append(math.log(route_params[name]["A_bytes"]))

    X = np.array(X_list)
    logP = np.array(logP_list)
    logA = np.array(logA_list)

    w_P = solve(X, logP)
    w_A = solve(X, logA)

    pred_logP = X @ w_P
    pred_logA = X @ w_A
    resid_P = logP - pred_logP
    resid_A = logA - pred_logA

    # 安全包络：覆盖最大正残差（actual > predicted）。正残差意味着预测偏低
    # （P 被低估），外推时偏低更危险（会放行该拒的），所以往上膨胀。
    # 负残差（预测偏高）不触发膨胀——偏高只会多拒，是安全方向。
    envelope_P = max(float(np.max(resid_P)), 0.0)
    envelope_A = max(float(np.max(resid_A)), 0.0)

    # -- 外推到域外路由 --
    extrapolated: dict[str, dict] = {}
    withheld: dict[str, str] = {}

    for name in routes_out:
        if name in rejected_by_physical:
            withheld[name] = "物理门禁拒绝：加法形式在该路由不适用（P≈0）"
            continue
        mid = name.split("::")[0]
        geo = geo_by_model.get(mid, {})
        params = geo.get("language_parameters") or geo.get("total_parameters")
        if not params:
            withheld[name] = "无几何数据（language_parameters / total_parameters 缺失）"
            continue
        feat = features(name, geo)
        P_upper = math.exp(float(np.dot(feat, w_P)) + envelope_P)
        A_upper = math.exp(float(np.dot(feat, w_A)) + envelope_A)
        extrapolated[name] = {
            "P_bytes": P_upper, "A_bytes": A_upper,
            "P_gib": P_upper / GIB, "A_gib": A_upper / GIB,
        }

    # -- 找域外路由的 OOM 行，检查 false_admit_oom --
    all_oom = [r for r in table["rows"]
               if r.get("classification") == "oom" and r.get("total_tokens")]
    oom_out = [r for r in all_oom if route_of(r) in set(extrapolated)]

    false_admit_details = []
    false_admit_routes: set[str] = set()
    for row in oom_out:
        name = route_of(row)
        P = extrapolated[name]["P_bytes"]
        A = extrapolated[name]["A_bytes"]
        pred = P + A * row["mbs"] * (row["total_tokens"] / REF_TOKENS) ** delta
        upper = pred * math.exp(Z_P95 * sigma)
        if upper <= SAFE_LIMIT_BYTES:
            false_admit_details.append({
                "route": name,
                "tokens": row["total_tokens"],
                "mbs": row["mbs"],
                "upper_gib": upper / GIB,
                "safe_limit_gib": SAFE_LIMIT_BYTES / GIB,
            })
            false_admit_routes.add(name)

    # false admit 的路由 → 拒答
    for name in list(extrapolated):
        if name in false_admit_routes:
            withheld[name] = "外推后 false_admit_oom：上界放行了 OOM 行，拒答"
            del extrapolated[name]

    # 域内 OOM 验证（用原始 P/A，应该 false_admit=0）
    oom_in = [r for r in all_oom if route_of(r) in set(routes_in)]
    false_admit_in = 0
    for row in oom_in:
        name = route_of(row)
        P = route_params[name]["P_bytes"]
        A = route_params[name]["A_bytes"]
        pred = P + A * row["mbs"] * (row["total_tokens"] / REF_TOKENS) ** delta
        upper = pred * math.exp(Z_P95 * sigma)
        if upper <= SAFE_LIMIT_BYTES:
            false_admit_in += 1

    payload = {
        "schema": "sft_h800_vl_admission_extrapolation/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only",
        "source_admission": "h800_vl_admission_v2.json",
        "method": "Plan A: 几何回归 + 安全包络（max positive residual）",
        "form": "P_upper + A_upper * mbs * (tokens/4096)^delta, then * exp(1.645 * sigma)",
        "delta": delta,
        "sigma": sigma,
        "safe_limit_bytes": SAFE_LIMIT_BYTES,
        "P_model": {
            "feature_order": FEATURE_ORDER,
            "coefficients": [float(c) for c in w_P],
            "params_source": "language_parameters if present else total_parameters",
            "fitted_on_routes": len(X_list),
            "residual_range_log": [float(np.min(resid_P)), float(np.max(resid_P))],
            "max_positive_residual_log": envelope_P,
            "safety_envelope": "P_upper = exp(log(P_pred) + max_positive_residual)",
        },
        "A_model": {
            "feature_order": FEATURE_ORDER,
            "coefficients": [float(c) for c in w_A],
            "params_source": "language_parameters if present else total_parameters",
            "fitted_on_routes": len(X_list),
            "residual_range_log": [float(np.min(resid_A)), float(np.max(resid_A))],
            "max_positive_residual_log": envelope_A,
            "safety_envelope": "A_upper = exp(log(A_pred) + max_positive_residual)",
        },
        "extrapolated_routes": {
            name: {"P_bytes": d["P_bytes"], "A_bytes": d["A_bytes"],
                   "P_gib": d["P_gib"], "A_gib": d["A_gib"]}
            for name, d in sorted(extrapolated.items())
        },
        "withheld_routes": {k: v for k, v in sorted(withheld.items())},
        "false_admit_oom_check": {
            "in_domain": {
                "oom_rows": len(oom_in),
                "false_admit": false_admit_in,
                "method": "原始 P/A from admission_v2（验证用）",
            },
            "out_of_domain": {
                "oom_rows": len(oom_out),
                "false_admit_before_withhold": len(false_admit_details),
                "false_admit_after_withhold": 0,
                "details": false_admit_details,
            },
        },
        "extrapolation_boundary": {
            "routes_out_total": len(routes_out),
            "routes_extrapolated": len(extrapolated),
            "routes_withheld": len(withheld),
        },
    }

    with OUT.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    # -- 打印 --
    print("=== 准入外推 ===")
    print(f"  源：admission_v2（{len(routes_in)} 条路由，delta={delta:.4f} sigma={sigma:.4f}）")
    print(f"  域外：{len(routes_out)} 条")
    print(f"  外推：{len(extrapolated)} 条")
    print(f"  拒答：{len(withheld)} 条")
    if skipped:
        print(f"  跳过（无几何）：{len(skipped)} 条")
    print()
    print("=== P 回归 ===")
    print(f"  系数: {[round(c, 6) for c in w_P]}")
    print(f"  残差范围: [{np.min(resid_P):.4f}, {np.max(resid_P):.4f}]")
    print(f"  安全包络: +{envelope_P:.4f}（log 空间）")
    print()
    print("=== A 回归 ===")
    print(f"  系数: {[round(c, 6) for c in w_A]}")
    print(f"  残差范围: [{np.min(resid_A):.4f}, {np.max(resid_A):.4f}]")
    print(f"  安全包络: +{envelope_A:.4f}（log 空间）")
    print()
    print("=== false_admit_oom 检查 ===")
    print(f"  域内 OOM: {len(oom_in)} 行, false_admit = {false_admit_in}")
    print(f"  域外 OOM: {len(oom_out)} 行, false_admit = {len(false_admit_details)}（已拒答对应路由）")
    if false_admit_details:
        for d in false_admit_details:
            print(f"    {d['route']}  tokens={d['tokens']} mbs={d['mbs']}  upper={d['upper_gib']:.1f}G")
    print()
    print("=== 外推路由 ===")
    for name, d in sorted(extrapolated.items()):
        print(f"  {name:40s}  P={d['P_gib']:6.1f}G  A={d['A_gib']:6.1f}G")
    print()
    print("=== 拒答路由 ===")
    for name, reason in sorted(withheld.items()):
        print(f"  {name:40s}  {reason}")
    print()
    print(f"产物 -> {OUT}")


def route_of(row: dict) -> str:
    return "%s::g%s_z%s::gc%d" % (row["model_id"], row["gpu_count"],
                                 row["zero_stage"], int(bool(row["gc"])))


if __name__ == "__main__":
    main()
