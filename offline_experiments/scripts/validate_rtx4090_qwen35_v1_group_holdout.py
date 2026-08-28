#!/usr/bin/env python3
"""对 4090 Qwen3.5 v1 做 leave-group-out 交叉验证。

为什么必须做：VL 侧的教训是 LOO 会系统性低估真实泛化误差——VL v1 的 LOO
吞吐 MAPE 是 18.6%，但真 out-of-sample 是 56.3%。原因是 LOO 每次只留一行，
同配置的邻居仍在训练集里，等于"抄答案"。留出整组才能暴露特征之间的隐藏耦合。

本脚本按三个结构轴分别留出：
  1. 按模型（0.8B / 4B / 9B）——检验跨模型规模外推
  2. 按 pack_mode（off / neat / pack）——检验 packing 机制外推
  3. 按注意力后端（fa2 / sdpa）——检验 regime 外推
  4. 按数据集——检验序列长度分布外推
"""
from __future__ import annotations

import numpy as np

import fit_rtx4090_qwen35_v1 as F

ALPHAS_MEM = [0.1, 1, 3, 10, 30, 100, 300, 1000]
ALPHAS_TPS = [0.01, 0.1, 1, 3, 10, 30, 100]


def _regime(r):
    pm = r.get("pack_mode", "off")
    if pm in ("neat", "pack"):
        return f"pack:{pm}"
    return f"nopack:{r['attn']}"


AXES = {
    "模型规模": lambda r: r["model"],
    "packing 模式": lambda r: r.get("pack_mode", "off"),
    "注意力后端": lambda r: r["attn"],
    "数据集": lambda r: r["dataset"],
    "regime": _regime,
}


def evaluate_holdout(train_ok, hold_ok, hold_oom):
    """在 train_ok 上拟合，在留出组上评估。返回指标字典。"""
    if len(train_ok) < 20:
        return None
    Xt = np.asarray([F.memory_features(r)[0] for r in train_ok])
    yt = np.asarray([float(r["peak_mib"]) for r in train_ok])
    w, alpha = F.ridge_fit(Xt, yt, ALPHAS_MEM)
    resid = yt - Xt @ w
    safety = float(max(np.quantile(np.maximum(resid, 0), 0.95), 0.0))

    tps_train = [r for r in train_ok if r.get("tokens_per_s") and r["tokens_per_s"] > 0]
    Xs = np.asarray([F.throughput_features(r)[0] for r in tps_train])
    ys = np.asarray([np.log(float(r["tokens_per_s"])) for r in tps_train])
    ws, _ = F.ridge_fit(Xs, ys, ALPHAS_TPS)

    out = dict(n_train=len(train_ok), safety=safety, alpha=alpha)
    # 显存 + 吞吐（成功行）
    if hold_ok:
        preds, acts, tp_rel, fr = [], [], [], 0
        for r in hold_ok:
            f, _ = F.memory_features(r)
            c = float(f @ w)
            preds.append(c); acts.append(float(r["peak_mib"]))
            if c + safety + 800 >= F.CARD_MEMORY_MIB and r["peak_mib"] < 22000:
                fr += 1
            if r.get("tokens_per_s") and r["tokens_per_s"] > 0:
                g, _ = F.throughput_features(r)
                tp_rel.append((float(np.exp(g @ ws)) - r["tokens_per_s"]) / r["tokens_per_s"])
        preds = np.asarray(preds); acts = np.asarray(acts)
        out.update(
            ok_n=len(preds),
            mem_rmse=float(np.sqrt(np.mean((preds - acts) ** 2))),
            mem_rel_mae=float(np.mean(np.abs(preds - acts) / acts)),
            mem_bias=float(np.mean(preds - acts)),
            false_reject=fr,
        )
        if tp_rel:
            tp_rel = np.asarray(tp_rel)
            out.update(tps_n=len(tp_rel),
                       tps_mape=float(np.mean(np.abs(tp_rel))),
                       tps_bias=float(np.mean(tp_rel)))
    else:
        out["ok_n"] = 0
    # OOM 召回
    if hold_oom:
        tp = 0
        for r in hold_oom:
            f, _ = F.memory_features(r)
            if float(f @ w) + safety + 800 >= F.CARD_MEMORY_MIB:
                tp += 1
        out.update(oom_n=len(hold_oom), oom_tp=tp, oom_recall=tp / len(hold_oom))
    else:
        out["oom_n"] = 0
    return out


def main():
    rows = F.load_observations()
    feasible = [r for r in rows if not F.is_infeasible(r)]
    ok = [r for r in feasible if r["exit"] == 0 and not r["oom"] and r.get("peak_mib")]
    oom = [r for r in feasible if r["oom"]]
    print(f"可行行：成功 {len(ok)}  OOM {len(oom)}\n")

    grand = dict(fr=0, oom_tp=0, oom_n=0, tps_abs=[], mem_rel=[])
    for axis_name, keyfn in AXES.items():
        groups = sorted({keyfn(r) for r in ok + oom})
        print(f"########## 按{axis_name}留出（{len(groups)} 组）##########")
        hdr = (f"{'留出组':22} {'训练n':>5} {'ok_n':>4} {'oom_n':>5} "
               f"{'显存RMSE':>8} {'相对MAE':>7} {'偏差':>7} {'误拒':>4} "
               f"{'吞吐MAPE':>8} {'吞吐偏差':>8} {'OOM召回':>9}")
        print(hdr); print("-" * len(hdr))
        for gname in groups:
            h_ok = [r for r in ok if keyfn(r) == gname]
            h_oom = [r for r in oom if keyfn(r) == gname]
            train = [r for r in ok if keyfn(r) != gname]
            res = evaluate_holdout(train, h_ok, h_oom)
            if res is None:
                print(f"{str(gname):22} 跳过（训练样本 {len(train)} 太少）")
                continue
            mr = f"{res['mem_rmse']:8.0f}" if res["ok_n"] else "     n/a"
            rm = f"{res['mem_rel_mae']*100:6.1f}%" if res["ok_n"] else "    n/a"
            bi = f"{res['mem_bias']:+7.0f}" if res["ok_n"] else "    n/a"
            fr = f"{res['false_reject']:4d}" if res["ok_n"] else " n/a"
            tm = f"{res['tps_mape']*100:7.1f}%" if res.get("tps_n") else "     n/a"
            tb = f"{res['tps_bias']*100:+7.1f}%" if res.get("tps_n") else "     n/a"
            rc = (f"{res['oom_tp']:3d}/{res['oom_n']:<3d}={res['oom_recall']*100:3.0f}%"
                  if res["oom_n"] else "      n/a")
            print(f"{str(gname):22} {res['n_train']:>5} {res['ok_n']:>4} {res['oom_n']:>5} "
                  f"{mr} {rm} {bi} {fr} {tm} {tb} {rc}")
            grand["fr"] += res.get("false_reject", 0)
            grand["oom_tp"] += res.get("oom_tp", 0)
            grand["oom_n"] += res["oom_n"]
            if res.get("tps_n"):
                grand["tps_abs"].append(res["tps_mape"])
            if res["ok_n"]:
                grand["mem_rel"].append(res["mem_rel_mae"])
        print()

    print("========== 跨全部留出组的汇总 ==========")
    print(f"  OOM 召回 = {grand['oom_tp']}/{grand['oom_n']} "
          f"= {grand['oom_tp']/max(grand['oom_n'],1)*100:.1f}%")
    print(f"  ok 误拒累计 = {grand['fr']}")
    if grand["mem_rel"]:
        print(f"  显存相对 MAE：中位 {np.median(grand['mem_rel'])*100:.1f}%  "
              f"最差 {max(grand['mem_rel'])*100:.1f}%")
    if grand["tps_abs"]:
        print(f"  吞吐 MAPE：中位 {np.median(grand['tps_abs'])*100:.1f}%  "
              f"最差 {max(grand['tps_abs'])*100:.1f}%")


if __name__ == "__main__":
    main()
