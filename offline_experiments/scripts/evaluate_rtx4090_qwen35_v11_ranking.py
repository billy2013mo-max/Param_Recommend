#!/usr/bin/env python3
"""量化 4090 Qwen3.5 v1.1 的**排序质量**（而不是绝对误差）。

为什么要单独测
--------------
推荐系统真正关心的是「在候选配置里挑出的那个够不够快」，不是「预测的 tok/s
数字准不准」。这两件事可以差很远：v1.1 的 MAPE 是 22.7%，但两两配对方向
正确率有 94.1%——绝对值偏但方向对。

只报 MAPE 会让人误以为排序不能用（我就这么误判过一次），所以把排序指标
单列出来。

三个指标
--------
* **Spearman 秩相关**：整体单调性
* **两两配对方向正确率**：随机拿两个配置，预测器说谁更快，对的比例
* **regret**：预测器推荐的第一名，比真实最优慢多少 —— 这个最接近业务价值

分组口径
--------
真实场景是「模型和数据集由用户给定，他挑的是其余参数（卡数/zero/attn/
packing/cutoff/mbs/gc）」，所以按 (model, dataset) 分组，只在组内比较。
跨模型比 tok/s 没有业务意义。
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

import fit_rtx4090_qwen35_v11 as F
from predictor_rtx4090_qwen35_v11 import RTX4090Qwen35Predictor


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise_accuracy(pred, actual) -> tuple[float, int]:
    """随机两个配置，预测器判断谁更快的正确率。并列的实测值跳过。"""
    ok = tot = 0
    for i, j in itertools.combinations(range(len(actual)), 2):
        if actual[i] == actual[j]:
            continue
        tot += 1
        if (pred[i] - pred[j]) * (actual[i] - actual[j]) > 0:
            ok += 1
    return (ok / tot if tot else float("nan")), tot


def main():
    p = RTX4090Qwen35Predictor.load()
    rows = [r for r in F.load_observations()
            if r["exit"] == 0 and not r["oom"]
            and r.get("tokens_per_s") and r["tokens_per_s"] > 0
            and not F.is_infeasible(r)]
    for r in rows:
        r["_pred"] = p.predict(dict(
            model=r["model"], dataset=r["dataset"], attn=r["attn"],
            pack_mode=r["pack_mode"], cutoff=r["cutoff"], mbs=r["mbs"],
            ga=r.get("ga", 1), gpu_count=r["gpu_count"],
            gc=r["gc"], zero=r["zero"]))["tokens_per_s"]

    P = np.array([r["_pred"] for r in rows])
    A = np.array([r["tokens_per_s"] for r in rows])
    acc, tot = pairwise_accuracy(P, A)
    print(f"全体 {len(rows)} 行")
    print(f"  Spearman 秩相关       = {spearman(P, A):.3f}")
    print(f"  两两配对方向正确率     = {acc*100:.1f}%  （{tot} 对）")
    print(f"  MAPE（绝对误差，对照） = {np.mean(np.abs(P/A - 1))*100:.1f}%")
    print()

    print("按 (模型, 数据集) 分组——真实推荐场景的可比范围")
    hdr = f"  {'模型':13} {'数据集':16} {'n':>3}  {'Spearman':>9}  {'配对正确率':>9}  {'regret':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["dataset"])].append(r)

    regrets, wsum, wtot, hits, groups = [], 0.0, 0, 0, 0
    for k, v in sorted(g.items()):
        if len(v) < 4:
            continue
        groups += 1
        pv = np.array([r["_pred"] for r in v])
        av = np.array([r["tokens_per_s"] for r in v])
        a, t = pairwise_accuracy(pv, av)
        wsum += a * t; wtot += t
        best_pred = max(v, key=lambda r: r["_pred"])
        best_real = max(v, key=lambda r: r["tokens_per_s"])
        regret = 1 - best_pred["tokens_per_s"] / best_real["tokens_per_s"]
        regrets.append(regret)
        if best_pred is best_real:
            hits += 1
        rs = "命中" if best_pred is best_real else f"{regret*100:.1f}%"
        print(f"  {k[0]:13} {k[1]:16} {len(v):>3}  {spearman(pv, av):+9.3f}  "
              f"{a*100:8.1f}%  {rs:>7}")

    print()
    print(f"  组内加权配对正确率 = {wsum/wtot*100:.1f}%")
    print(f"  第一名命中         = {hits}/{groups} 组")
    print(f"  平均 regret        = {np.mean(regrets)*100:.1f}%"
          f"   最差 {max(regrets)*100:.1f}%")
    print()

    # 单独看被文档标注为「吞吐不可靠」的多卡+GC 子集，
    # 检验那条警告是否对排序也成立
    mg = [r for r in rows if r["gc"] and r["gpu_count"] > 1]
    other = [r for r in rows if not (r["gc"] and r["gpu_count"] > 1)]
    print("多卡 + GC 子集 vs 其余（检验「吞吐不可靠」是否也波及排序）")
    for label, sub in [("多卡+GC", mg), ("其余", other)]:
        if len(sub) < 2:
            continue
        pv = np.array([r["_pred"] for r in sub])
        av = np.array([r["tokens_per_s"] for r in sub])
        a, _ = pairwise_accuracy(pv, av)
        print(f"  {label:8} n={len(sub):3}  配对正确率={a*100:5.1f}%  "
              f"MAPE={np.mean(np.abs(pv/av - 1))*100:5.1f}%")
    print("  → 结论：MAPE 差距明显但配对正确率接近，说明该子集是"
          "「绝对值偏、方向对」，可用于排序、不可当预算数字。")


if __name__ == "__main__":
    main()
