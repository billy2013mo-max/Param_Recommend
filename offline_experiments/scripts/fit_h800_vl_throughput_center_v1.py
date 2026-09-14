#!/usr/bin/env python3
"""VL 吞吐模型：分路由拟合 + 两种留出验收。

吞吐数据怎么来的
----------------
609 行里只有 270 行带 `effective_tokens_per_second` 字段，多卡那两批（route_fill
288 格、multigpu 40 格）绝大多数是 None。但它们的 token 账本和测量秒数都在，
可以反算：

    tok/s = measured_totals.effective_tokens / measured_seconds

口径在有字段的行上逐一验证过：对照行记录值 789.0，反算 789.0，完全一致。
反算出来的行标 `tps_source = derived_from_ledger`，与直接读到的区分开。

注意这是**单 rank 吞吐**（每卡处理自己那份 token 的速率）。多卡的总吞吐是它乘
卡数，但建模用单 rank——推荐器排序时关心的是同一配置下的相对快慢，而卡数已经
是路由的一部分。

路由与模型形式
--------------
沿用显存模型 V4 的路由定义 (模型, 卡数, ZeRO, gc)。路由内：

    log(tok/s) = a + b·log(total_tokens) + c·log2(mbs) + d·visual_share

吞吐这里 `share` 对 gc 开关都放：视觉 token 要多走一趟 ViT 前向，那是实打实的
计算时间，与 GC 是否丢弃语言激活无关（GC 只影响显存和重算，不改变 ViT 那趟）。
这跟显存模型不同——显存里 gc 关时视觉部分被语言激活盖住，吞吐里没有这种遮蔽。

验收怎么拆
----------
用户提出从现有数据拆验收集。随机拆不行——同一路由的重复行会被分到两边，验收就
不是分布外的了。所以用两种结构化留出：

    留出整条路由    检验能否外推到没见过的机制组合（回退到父路由或全局）
    留出整个长度档  检验路由内的长度响应能否外推到没测过的长度

另外保留 prospective_v3 那 27 行作为来源隔离验收——它是当初就设计成前瞻验收的，
且吞吐在 V3 里一次都没被预测过（全部返回"超出支持域"）。
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
OUT = REPO / "artifacts" / "h800_vl_throughput_center_v1.json"

MIN_ROWS = 8
MIN_LENGTH_SPAN = 3.0


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def throughput_of(job_id: str) -> tuple[float, str] | tuple[None, None]:
    """取单 rank 有效吞吐。字段缺失时从 token 账本反算。"""
    import glob
    import os

    attempts = glob.glob(str(REPO / "results" / job_id / "attempts" / "*"))
    if not attempts:
        return None, None
    newest = max(attempts, key=os.path.getmtime)
    direct, derived = [], []
    for path in glob.glob(os.path.join(newest, "metrics", "summary.rank*.json")):
        summary = read_json(Path(path))
        value = summary.get("effective_tokens_per_second")
        if isinstance(value, (int, float)) and value > 0:
            direct.append(float(value))
            continue
        ledger = (summary.get("token_ledger_evidence") or {}).get("measured_totals") or {}
        tokens = ledger.get("effective_tokens")
        seconds = summary.get("measured_seconds")
        if tokens and seconds and seconds > 0:
            derived.append(tokens / seconds)
    if direct:
        return sum(direct) / len(direct), "summary_field"
    if derived:
        return sum(derived) / len(derived), "derived_from_ledger"
    return None, None


def route_of(row: dict) -> str:
    return (f"{row['model_id']}::g{row['gpu_count']}"
            f"_z{row['zero_stage']}::gc{int(bool(row['gc']))}")


def parent_route_of(row: dict) -> str:
    return f"{row['model_id']}::g{row['gpu_count']}_z{row['zero_stage']}"


def features(row: dict) -> list[float]:
    return [1.0, math.log(row["total_tokens"]), math.log2(row["mbs"]),
            row["visual_tokens"] / row["total_tokens"]]


def global_features(row: dict) -> list[float]:
    params = row["language_parameters"] or row["total_parameters"]
    return [1.0, math.log(params), math.log(row["total_tokens"]),
            math.log2(row["mbs"]), 1.0 if row["gc"] else 0.0,
            math.log2(row["gpu_count"]),
            1.0 if row["zero_stage"] == 2 else 0.0,
            1.0 if row["zero_stage"] == 3 else 0.0,
            row["visual_tokens"] / row["total_tokens"]]


def solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.solve(matrix.T @ matrix + 1e-8 * np.eye(matrix.shape[1]),
                           matrix.T @ target)


def fittable(rows: list[dict], parameters: int = 4) -> bool:
    if len(rows) < max(MIN_ROWS, parameters + 4):
        return False
    lengths = [r["total_tokens"] for r in rows]
    return max(lengths) / min(lengths) >= MIN_LENGTH_SPAN


class RoutedThroughput:
    def __init__(self, train: list[dict]):
        self.routes: dict[str, np.ndarray] = {}
        self.parents: dict[str, np.ndarray] = {}
        self.route_rows: dict[str, int] = {}

        by_route, by_parent = defaultdict(list), defaultdict(list)
        for row in train:
            by_route[route_of(row)].append(row)
            by_parent[parent_route_of(row)].append(row)

        for name, rows in by_route.items():
            self.route_rows[name] = len(rows)
            if fittable(rows):
                x = np.array([features(r) for r in rows])
                y = np.log([r["tokens_per_second"] for r in rows])
                self.routes[name] = solve(x, y)
        for name, rows in by_parent.items():
            if fittable(rows):
                x = np.array([features(r) for r in rows])
                y = np.log([r["tokens_per_second"] for r in rows])
                self.parents[name] = solve(x, y)

        x = np.array([global_features(r) for r in train])
        y = np.log([r["tokens_per_second"] for r in train])
        self.global_weights = solve(x, y)

    def predict_one(self, row: dict) -> tuple[float, str]:
        name = route_of(row)
        if name in self.routes:
            return float(np.exp(np.dot(features(row), self.routes[name]))), "route"
        parent = parent_route_of(row)
        if parent in self.parents:
            return float(np.exp(np.dot(features(row), self.parents[parent]))), "parent"
        return float(np.exp(np.dot(global_features(row), self.global_weights))), "global"

    def evaluate(self, rows: list[dict]) -> dict:
        errors, levels = [], []
        for row in rows:
            predicted, level = self.predict_one(row)
            errors.append((predicted - row["tokens_per_second"]) / row["tokens_per_second"])
            levels.append(level)
        errors = np.array(errors)
        by_level = defaultdict(list)
        for error, level in zip(errors, levels):
            by_level[level].append(abs(error))
        return {
            "rows": len(rows),
            "mape": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "p90": float(np.percentile(np.abs(errors), 90)),
            "max_abs": float(np.max(np.abs(errors))),
            "by_fit_level": {k: {"rows": len(v), "mape": float(np.mean(v))}
                             for k, v in sorted(by_level.items())},
        }


def ranking_quality(model: RoutedThroughput, rows: list[dict]) -> dict:
    """排序质量：推荐器真正关心的是「同一工作量下哪个机制更快」。

    按 (模型, 工作量) 分组，组内按预测与实测各自排序，看成对排序准确率与
    Top-1 regret（选中的那个比真最优慢多少）。
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_id"], round(row["total_tokens"]))].append(row)

    pairs = correct = 0
    regrets = []
    for members in groups.values():
        if len(members) < 2:
            continue
        predicted = {id(r): model.predict_one(r)[0] for r in members}
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pairs += 1
                if ((predicted[id(a)] > predicted[id(b)])
                        == (a["tokens_per_second"] > b["tokens_per_second"])):
                    correct += 1
        best_predicted = max(members, key=lambda r: predicted[id(r)])
        best_actual = max(members, key=lambda r: r["tokens_per_second"])
        regrets.append(1.0 - best_predicted["tokens_per_second"]
                       / best_actual["tokens_per_second"])
    return {
        "informative_groups": sum(1 for m in groups.values() if len(m) >= 2),
        "pairs": pairs,
        "pairwise_accuracy": correct / pairs if pairs else None,
        "mean_top1_regret": float(np.mean(regrets)) if regrets else None,
        "worst_top1_regret": float(np.max(regrets)) if regrets else None,
    }


def leave_one_route_out(rows: list[dict]) -> dict:
    """留出整条路由：检验能否外推到没见过的机制组合。"""
    by_route = defaultdict(list)
    for row in rows:
        by_route[route_of(row)].append(row)
    errors, levels = [], []
    for held, held_rows in by_route.items():
        train = [r for r in rows if route_of(r) != held]
        if len(train) < 40:
            continue
        model = RoutedThroughput(train)
        for row in held_rows:
            predicted, level = model.predict_one(row)
            errors.append(abs(predicted - row["tokens_per_second"])
                          / row["tokens_per_second"])
            levels.append(level)
    from collections import Counter
    return {"held_out_rows": len(errors),
            "mape": float(np.mean(errors)) if errors else None,
            "p90": float(np.percentile(errors, 90)) if errors else None,
            "fit_levels": dict(Counter(levels))}


def leave_one_length_out(rows: list[dict]) -> dict:
    """留出整个长度档：检验路由内长度响应能否外推。"""
    by_route = defaultdict(list)
    for row in rows:
        by_route[route_of(row)].append(row)
    errors = []
    for members in by_route.values():
        if not fittable(members):
            continue
        lengths = sorted({round(r["total_tokens"]) for r in members})
        if len(lengths) < 6:
            continue
        for held in lengths:
            fit_rows = [r for r in members if round(r["total_tokens"]) != held]
            test_rows = [r for r in members if round(r["total_tokens"]) == held]
            if not fittable(fit_rows):
                continue
            x = np.array([features(r) for r in fit_rows])
            y = np.log([r["tokens_per_second"] for r in fit_rows])
            weights = solve(x, y)
            for row in test_rows:
                predicted = math.exp(float(np.dot(features(row), weights)))
                errors.append(abs(predicted - row["tokens_per_second"])
                              / row["tokens_per_second"])
    return {"held_out_points": len(errors),
            "mape": float(np.mean(errors)) if errors else None,
            "p90": float(np.percentile(errors, 90)) if errors else None}


def main() -> None:
    table = read_json(TABLE)
    rows = []
    sources = defaultdict(int)
    for row in table["rows"]:
        if not row.get("total_tokens") or not row.get("visual_tokens"):
            continue
        value, origin = throughput_of(row["job_id"])
        if value is None:
            continue
        rows.append({**row, "tokens_per_second": value, "tps_source": origin})
        sources[origin] += 1

    train = [r for r in rows if r["source"] != "prospective_v3"]
    test = [r for r in rows if r["source"] == "prospective_v3"]

    model = RoutedThroughput(train)
    payload = {
        "schema": "sft_h800_vl_throughput_center/v1_routed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only",
        "production_ranking_allowed": False,
        "target": "effective_tokens_per_second（单 rank）",
        "target_note": "多卡总吞吐 = 单 rank × 卡数；建模用单 rank，"
                       "因为卡数已是路由的一部分",
        "route_definition": "(model_id, gpu_count, zero_stage, gc)",
        "within_route_form": "log(tok/s) = a + b·log(tokens) + c·log2(mbs) + d·share",
        "why_share_always": "视觉 token 要多走一趟 ViT 前向，那是实打实的计算时间，"
                            "与 GC 是否丢弃语言激活无关；显存模型里 gc 关时视觉被"
                            "语言激活遮蔽，吞吐里没有这种遮蔽",
        "rows_total": len(rows),
        "tps_sources": dict(sources),
        "routes_fitted": len(model.routes),
        "route_coefficients": {
            name: {"rows": model.route_rows[name], "intercept": float(w[0]),
                   "log_tokens": float(w[1]), "log_mbs": float(w[2]),
                   "share": float(w[3])}
            for name, w in sorted(model.routes.items())
        },
        "train": model.evaluate(train),
        "acceptance_source_isolated": {
            "test_source": "prospective_v3",
            "note": "该批吞吐在 V3 验收里全部返回「超出支持域」，从未被预测过",
            **model.evaluate(test),
        },
        "acceptance_leave_one_route_out": leave_one_route_out(rows),
        "acceptance_leave_one_length_out": leave_one_length_out(rows),
        "ranking_train": ranking_quality(model, train),
        "ranking_test": ranking_quality(model, test),
    }
    with OUT.open("w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    print(f"可用行 {len(rows)}（字段直读 {sources['summary_field']}，"
          f"账本反算 {sources['derived_from_ledger']}）")
    print(f"独立拟合路由 {len(model.routes)} 条\n")
    print(f"{'':26}{'行数':>6}{'MAPE':>9}{'偏差':>9}{'P90':>8}{'最大':>8}")
    for label, block in (("训练集", payload["train"]),
                         ("验收（来源隔离）", payload["acceptance_source_isolated"])):
        print(f"{label:<24}{block['rows']:>6}{block['mape']*100:>8.1f}%"
              f"{block['bias']*100:>+8.1f}%{block['p90']*100:>7.1f}%{block['max_abs']*100:>7.1f}%")
    for label, key in (("验收（留出整条路由）", "acceptance_leave_one_route_out"),
                       ("验收（留出整个长度档）", "acceptance_leave_one_length_out")):
        block = payload[key]
        n = block.get("held_out_rows") or block.get("held_out_points")
        if block["mape"] is not None:
            print(f"{label:<24}{n:>6}{block['mape']*100:>8.1f}%{'':>9}"
                  f"{block['p90']*100:>7.1f}%")
    print("\n=== 排序质量（推荐器真正关心的）===")
    for label, key in (("训练集", "ranking_train"), ("验收集", "ranking_test")):
        block = payload[key]
        if block["pairwise_accuracy"] is None:
            print(f"  {label}: 无可比对比组")
            continue
        print(f"  {label}: {block['informative_groups']} 组 / {block['pairs']} 对  "
              f"成对准确率 {block['pairwise_accuracy']:.1%}  "
              f"平均 Top-1 regret {block['mean_top1_regret']:.1%}  "
              f"最差 {block['worst_top1_regret']:.1%}")
    print(f"\n产物 → {OUT}")


if __name__ == "__main__":
    main()
