#!/usr/bin/env python3
"""对 v1 预测器做 leave-(model,dataset)-out 交叉验证。

每次留出一个 (model, dataset) 组的所有行（ok 用作显存回归 & 吞吐；oom 用作准入
召回），在其余 7 组上重新拟合，然后在留出组上评估：

- ok 行：peak_center 与实测差距、admit=True 是否正确
- oom 行：admit=False 是否正确

输出一张 per-group 结果表 + 全局汇总。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 复用拟合脚本里的特征与 ridge
import fit_rtx4090_vl_v1 as fit

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "artifacts" / "rtx4090_vl_observations_20260823" / "matrix_results_fixed.jsonl"


def _prepare(rows):
    """把 VL 行拆成 ok / oom，返回 (ok_X, ok_y, ok_rows), (oom_rows)。"""
    vl = [r for r in rows if r.get("track") != "text_smoke"]
    ok = [r for r in vl if r["exit"] == 0 and not r["oom"] and r.get("peak_mib")]
    oom = [r for r in vl if r["oom"]]
    ok_X, ok_y, names = [], [], None
    for r in ok:
        f, n = fit.memory_features(r)
        ok_X.append(f); ok_y.append(float(r["peak_mib"]))
        if names is None: names = n
    return (np.asarray(ok_X), np.asarray(ok_y), ok, names, oom)


def _fit_and_predict(train_rows, holdout_ok, holdout_oom, names):
    train_X = np.asarray([fit.memory_features(r)[0] for r in train_rows])
    train_y = np.asarray([float(r["peak_mib"]) for r in train_rows])
    w, alpha = fit.ridge_fit(train_X, train_y, alphas=[0.1, 1, 3, 10, 30, 100, 300])
    train_pred = train_X @ w
    train_resid = train_y - train_pred
    safety = float(max(np.quantile(np.maximum(train_resid, 0), 0.95), 0.0))
    # 吞吐用 log 空间
    tp_train = [r for r in train_rows if r.get("tokens_per_s") and r["tokens_per_s"] > 0]
    tp_X = np.asarray([fit.throughput_features(r)[0] for r in tp_train])
    tp_y = np.asarray([np.log(float(r["tokens_per_s"])) for r in tp_train])
    tp_w, _ = fit.ridge_fit(tp_X, tp_y, alphas=[0.01, 0.1, 1, 3, 10, 30])

    # 评估留出组
    result = dict(alpha=alpha, safety=safety, n_train=len(train_rows))
    if holdout_ok:
        preds, actuals, tp_preds, tp_actuals, false_reject = [], [], [], [], 0
        for r in holdout_ok:
            f, _ = fit.memory_features(r)
            center = float(f @ w); upper = center + safety
            preds.append(center); actuals.append(float(r["peak_mib"]))
            if upper + 800 >= fit.CARD_MEMORY_MIB and r["peak_mib"] < 22000:
                false_reject += 1
            if r.get("tokens_per_s") and r["tokens_per_s"] > 0:
                g, _ = fit.throughput_features(r)
                tp_preds.append(float(np.exp(g @ tp_w)))
                tp_actuals.append(float(r["tokens_per_s"]))
        preds = np.asarray(preds); actuals = np.asarray(actuals)
        result["ok_n"] = len(preds)
        result["ok_mem_rmse"] = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        result["ok_mem_mae"] = float(np.mean(np.abs(preds - actuals)))
        result["ok_false_reject"] = false_reject
        if tp_preds:
            tp_preds = np.asarray(tp_preds); tp_actuals = np.asarray(tp_actuals)
            result["ok_tps_mape"] = float(np.mean(np.abs(tp_preds - tp_actuals) / tp_actuals))
            result["ok_tps_n"] = len(tp_preds)
    else:
        result["ok_n"] = 0
    # OOM 召回
    if holdout_oom:
        tp = 0
        for r in holdout_oom:
            f, _ = fit.memory_features(r)
            upper = float(f @ w) + safety
            if upper + 800 >= fit.CARD_MEMORY_MIB:
                tp += 1
        result["oom_n"] = len(holdout_oom); result["oom_recall"] = tp / len(holdout_oom)
        result["oom_tp"] = tp
    else:
        result["oom_n"] = 0
    return result


def main():
    rows = [json.loads(l) for l in open(DATA)]
    _, _, ok, names, oom = _prepare(rows)

    # 所有 (model, dataset) 组：以 ok+oom 的并集为准
    groups = sorted({(r["model"], r["dataset"]) for r in ok + oom})

    print(f"总 ok={len(ok)} oom={len(oom)}  分成 {len(groups)} 组做 leave-group-out\n")
    header = f"{'group':30}  {'ok_n':>4}  {'oom_n':>4}  {'RMSE':>6}  {'MAE':>6}  {'FR':>2}  {'MAPE':>5}  {'oom_recall':>10}  {'safety':>6}"
    print(header)
    print("-" * len(header))

    agg = dict(ok_pred=[], ok_act=[], tp_pred=[], tp_act=[], oom_admit=0, oom_total=0, fr=0)
    for g in groups:
        heldout_ok = [r for r in ok if (r["model"], r["dataset"]) == g]
        heldout_oom = [r for r in oom if (r["model"], r["dataset"]) == g]
        train = [r for r in ok if (r["model"], r["dataset"]) != g]
        if len(train) < 8:
            print(f"{g!s:30}  跳过（训练样本 {len(train)} 太少）"); continue
        res = _fit_and_predict(train, heldout_ok, heldout_oom, names)
        ok_str = f"{res.get('ok_mem_rmse', float('nan')):6.0f}" if res["ok_n"] else "   n/a"
        mae_str = f"{res.get('ok_mem_mae', float('nan')):6.0f}" if res["ok_n"] else "   n/a"
        fr_str = f"{res.get('ok_false_reject', 0):2d}" if res["ok_n"] else "n/a"
        mape_str = f"{res.get('ok_tps_mape', float('nan'))*100:5.1f}" if res.get("ok_tps_n") else "  n/a"
        rec_str = (f"{res['oom_tp']:2d}/{res['oom_n']:<2d}={res['oom_recall']*100:5.1f}%"
                   if res["oom_n"] else "     n/a  ")
        print(f"{g!s:30}  {res['ok_n']:4d}  {res['oom_n']:4d}  {ok_str}  {mae_str}  {fr_str}  {mape_str}  {rec_str}  {res['safety']:6.0f}")

        # 汇总
        if res["ok_n"]:
            for r in heldout_ok:
                f, _ = fit.memory_features(r); pass
        agg["fr"] += res.get("ok_false_reject", 0)
        agg["oom_admit"] += res.get("oom_tp", 0)
        agg["oom_total"] += res["oom_n"]

    print()
    print(f"跨组汇总：错拒 ok 行 = {agg['fr']}，OOM 召回 = {agg['oom_admit']}/{agg['oom_total']} "
          f"= {agg['oom_admit']/max(agg['oom_total'],1)*100:.1f}%")


if __name__ == "__main__":
    main()
