#!/usr/bin/env python3
"""VL 显存模型 V5：重拟合中心 + 用 OOM 右删失下界定准入上界。

相对 V4 的两处变化
------------------
一、多卡路由的 log_mbs 现在可辨识了。V4 时 16 个多卡路由的 mbs 恒为 1，
log_mbs 拟合成接近 0，模型会错认为「多卡下微批翻倍不增显存」。76 格 OOM
边界活动补了 mbs 属于 {2,4,8,16}，这一项终于有数据支撑。

二、有 27 个 OOM 右删失下界了。此前 609 行 OOM 为 0，准入上界定不出来
（用 P95x1.15 算出 3.59 倍，宽到不会拒绝任何配置）。现在可以用真实的 OOM
位置来标定上界。

准入上界怎么定
--------------
上界 = 中心预测 x 倍率。倍率要同时满足两条方向相反的约束：

    覆盖：success 行的 reserved 必须小于等于上界（低了会低估真实需求）
    有效：OOM 行的上界必须高于安全线（低了会放行实际炸掉的配置）

扫倍率，取「错放 OOM = 0 且 success 覆盖率最高」的那个。这与 hybrid 那条线
同一套口径：中心负责「预测多少」，上界负责「按多少放行」。

OOM 行怎么用
------------
OOM 行没有可信峰值（崩溃时的读数不是真实需求量），只知道「真实需求 > 卡容量」，
是右删失下界。所以它不进中心拟合，只用来检验上界够不够高。
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "artifacts" / "h800_vl_memory_table_v1.json"
OUT = REPO / "artifacts" / "h800_vl_memory_center_v5.json"

MIN_ROWS = 8
MIN_LENGTH_SPAN = 3.0
CAPACITY_BYTES = 150142189568
SAFE_LIMIT_BYTES = 142635080089.6


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def route_of(row: dict) -> str:
    return (f"{row['model_id']}::g{row['gpu_count']}"
                f"_z{row['zero_stage']}::gc{int(bool(row['gc']))}")


def parent_route_of(row: dict) -> str:
    return f"{row['model_id']}::g{row['gpu_count']}_z{row['zero_stage']}"


def features(row: dict, with_share: bool) -> list[float]:
    base = [1.0, math.log(row["total_tokens"]), math.log2(row["mbs"])]
    if with_share:
        base.append(row["visual_tokens"] / row["total_tokens"])
    return base


def uses_share(rows: list[dict]) -> bool:
    """gc 开的路由才用 share，且占比在该路由内真的变化过。

    显存里 gc 关时视觉部分被语言激活遮蔽，等价性成立；gc 开时峰值落在视觉塔
    前向，视觉成为主导项。
    """
    if not rows or not rows[0]["gc"]:
        return False
    shares = [r["visual_tokens"] / r["total_tokens"] for r in rows]
    return max(shares) - min(shares) >= 0.2


def global_features(row: dict) -> list[float]:
    params = row["language_parameters"] or row["total_parameters"]
    gc = 1.0 if row["gc"] else 0.0
    share = row["visual_tokens"] / row["total_tokens"]
    return [
        1.0,
        math.log(params),
        math.log(row["total_tokens"]),
        math.log2(row["mbs"]),
        gc,
        math.log2(row["gpu_count"]),
        1.0 if row["zero_stage"] == 2 else 0.0,
        1.0 if row["zero_stage"] == 3 else 0.0,
        share * gc,
    ]


def solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    ridge = 1e-8 * np.eye(matrix.shape[1])
    return np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ target)


def fittable(rows: list[dict], parameters: int) -> bool:
    if len(rows) < max(MIN_ROWS, parameters + 4):
        return False
    lengths = [r["total_tokens"] for r in rows]
    return max(lengths) / min(lengths) >= MIN_LENGTH_SPAN


class RoutedCenter:
    def __init__(self, train: list[dict]):
        self.routes: dict[str, tuple[np.ndarray, bool]] = {}
        self.parents: dict[str, tuple[np.ndarray, bool]] = {}
        self.route_rows: dict[str, int] = {}
        self.mbs_levels: dict[str, int] = {}

        by_route: dict[str, list[dict]] = defaultdict(list)
        by_parent: dict[str, list[dict]] = defaultdict(list)
        for row in train:
            by_route[route_of(row)].append(row)
            by_parent[parent_route_of(row)].append(row)

        for name, rows in by_route.items():
            self.route_rows[name] = len(rows)
            self.mbs_levels[name] = len({int(r["mbs"]) for r in rows})
            share = uses_share(rows)
            if fittable(rows, 4 if share else 3):
                design = np.array([features(r, share) for r in rows])
                target = np.log([r["allocated_bytes"] for r in rows])
                self.routes[name] = (solve(design, target), share)

        for name, rows in by_parent.items():
            if fittable(rows, 3):
                design = np.array([features(r, False) for r in rows])
                target = np.log([r["allocated_bytes"] for r in rows])
                self.parents[name] = (solve(design, target), False)

        design = np.array([global_features(r) for r in train])
        target = np.log([r["allocated_bytes"] for r in train])
        self.global_weights = solve(design, target)

    def predict_one(self, row: dict) -> tuple[float, str]:
        name = route_of(row)
        if name in self.routes:
            weights, share = self.routes[name]
            return float(np.exp(np.dot(features(row, share), weights))), "route"
        parent = parent_route_of(row)
        if parent in self.parents:
            weights, share = self.parents[parent]
            return float(np.exp(np.dot(features(row, share), weights))), "parent"
        value = float(np.exp(np.dot(global_features(row), self.global_weights)))
        return value, "global"

    def evaluate(self, rows: list[dict]) -> dict:
        errors = []
        levels = []
        for row in rows:
            predicted, level = self.predict_one(row)
            actual = row["allocated_bytes"]
            errors.append((predicted - actual) / actual)
            levels.append(level)
        errors = np.array(errors)
        return {
            "rows": len(rows),
            "mape": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "p90": float(np.percentile(np.abs(errors), 90)),
            "max_abs": float(np.max(np.abs(errors))),
            "rows_over_50pct": int(np.sum(np.abs(errors) > 0.5)),
            "fit_levels": dict(Counter(levels)),
        }


def calibrate_upper(center, success: list[dict], oom: list[dict]) -> dict:
    """扫倍率，找「错放 OOM = 0 且 success 覆盖率最高」的那个。

    两条约束方向相反：倍率太低覆盖不住 success 的 reserved；太高则 OOM 行的
    上界仍落在安全线以下，模型会把该拒的配置放行。
    """
    candidates = [round(1.0 + 0.01 * i, 2) for i in range(301)]
    scan = []
    for multiplier in candidates:
        covered = 0
        for row in success:
            upper = center.predict_one(row)[0] * multiplier
            if row["reserved_bytes"] <= upper:
                covered += 1
        false_admit = 0
        for row in oom:
            upper = center.predict_one(row)[0] * multiplier
            if upper <= SAFE_LIMIT_BYTES:
                false_admit += 1
        scan.append({
            "multiplier": multiplier,
            "success_coverage": covered / len(success) if success else None,
            "false_admit_oom": false_admit,
        })

    feasible = [s for s in scan if s["false_admit_oom"] == 0]
    chosen = max(feasible, key=lambda s: s["success_coverage"]) if feasible else None
    fallback = min(scan, key=lambda s: (s["false_admit_oom"], -s["success_coverage"]))
    sample = [s for s in scan
              if abs(s["multiplier"] * 4 - round(s["multiplier"] * 4)) < 1e-9]
    return {
        "chosen": chosen,
        "best_effort": fallback,
        "zero_false_admit_possible": chosen is not None,
        "scan_range": [candidates[0], candidates[-1]],
        "sample": sample,
    }


def main() -> None:
    all_rows = read_json(TABLE)["rows"]
    fit_rows = [r for r in all_rows
                if r.get("allocated_bytes") and r.get("total_tokens")]
    oom_rows = [r for r in all_rows
                if r.get("classification") == "oom" and r.get("total_tokens")]

    train = [r for r in fit_rows if r["source"] != "prospective_v3"]
    test = [r for r in fit_rows if r["source"] == "prospective_v3"]

    center = RoutedCenter(train)
    success_with_reserved = [r for r in train if r.get("reserved_bytes")]
    upper = calibrate_upper(center, success_with_reserved, oom_rows)

    multi_gpu = {n: center.mbs_levels[n] for n in sorted(center.routes)
          if "g1_" not in n}
    log_mbs = {n: float(w[2]) for n, (w, _) in sorted(center.routes.items())}

    payload = {
        "schema": "sft_h800_vl_memory_center/v5_routed_with_admission",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only",
        "production_admission_allowed": False,
        "target": "allocated_bytes",
        "route_definition": "(model_id, gpu_count, zero_stage, gc)",
        "within_route_form": {
            "gc_off": "log(alloc) = a + b*log(tokens) + c*log2(mbs)",
            "gc_on": "log(alloc) = a + b*log(tokens) + c*log2(mbs) + d*visual_share",
        },
        "changes_since_v4": {
            "log_mbs_now_identifiable": "76 格 OOM 边界活动补了 mbs 属于 {2,4,8,16}；"
                  "V4 时多卡路由 mbs 恒为 1，log_mbs 拟合成接近 0",
            "oom_lower_bounds": f"{len(oom_rows)} 个右删失下界（此前为 0）",
        },
        "oom_rows_used_for": "仅用于标定准入上界；OOM 行没有可信峰值"
                 "（崩溃读数不是真实需求量），不进中心拟合",
        "routes_fitted": len(center.routes),
        "multi_gpu_mbs_levels": multi_gpu,
        "log_mbs_by_route": log_mbs,
        "route_coefficients": {
            name: {
                "rows": center.route_rows[name],
                "mbs_levels": center.mbs_levels[name],
                "uses_share": share,
                "intercept": float(weights[0]),
                "log_tokens": float(weights[1]),
                "log_mbs": float(weights[2]),
                **({"share": float(weights[3])} if share else {}),
            }
            for name, (weights, share) in sorted(center.routes.items())
        },
        "train": center.evaluate(train),
        "acceptance": {
            "test_source": "prospective_v3",
            "never_used_in_any_vl_fit": True,
            **center.evaluate(test),
        },
        "admission_upper": upper,
        "capacity_bytes": CAPACITY_BYTES,
        "safe_limit_bytes": SAFE_LIMIT_BYTES,
    }
    with OUT.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    print(f"训练 {len(train)} 行 / 评测 {len(test)} 行 / OOM 下界 {len(oom_rows)} 个")
    print(f"独立拟合路由 {len(center.routes)} 条")
    print("")
    print(f"{'':16}{'行数':>6}{'MAPE':>9}{'偏差':>9}{'P90':>8}{'最大':>8}{'>50%':>6}")
    for label, block in (("训练集", payload["train"]),
                  ("验收（分布外）", payload["acceptance"])):
        print(f"{label:<14}{block['rows']:>6}{block['mape']*100:>8.1f}%"
                  f"{block['bias']*100:>+8.1f}%{block['p90']*100:>7.1f}%"
                  f"{block['max_abs']*100:>7.1f}%{block['rows_over_50pct']:>6}")

    print("")
    print("=== 多卡路由的 log_mbs（V4 里这些都接近 0）===")
    for name in list(multi_gpu)[:8]:
        print(f"   {name:<36}mbs 档位 {multi_gpu[name]}   log_mbs {log_mbs[name]:+.4f}")

    print("")
    print("=== 准入上界标定 ===")
    if upper["zero_false_admit_possible"]:
        chosen = upper["chosen"]
        print(f"   倍率 {chosen['multiplier']:.2f}："
                  f"错放 OOM = {chosen['false_admit_oom']}，"
                  f"success 覆盖率 {chosen['success_coverage']:.1%}")
    else:
        best = upper["best_effort"]
        print(f"   没有倍率能做到零错放。最好的：x{best['multiplier']:.2f}，"
                  f"错放 {best['false_admit_oom']}，覆盖 {best['success_coverage']:.1%}")
    print("")
    print("   扫描样点：")
    for entry in upper["sample"]:
        print(f"  x{entry['multiplier']:.2f}   "
                  f"覆盖 {entry['success_coverage']:.1%}   "
                  f"错放 OOM {entry['false_admit_oom']}")
    print(f"")
    print(f"产物 -> {OUT}")


if __name__ == "__main__":
    main()
