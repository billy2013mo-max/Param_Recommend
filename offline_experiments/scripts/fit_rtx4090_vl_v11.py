#!/usr/bin/env python3
"""4090 VL v1.1 拟合：合并 v1 原始 108 行 + OOS 10 行 + 补数 ~22 行 → ~140 行。

相对 v1 的改动
--------------
* 数据合并：新增 out-of-sample 与补数矩阵结果
* 吞吐特征增加 `gc × log_gpu` 交叉项——OOS 揭示的具体缺陷
  （多卡 gc=on 预测偏快，单卡 gc=off 预测偏慢，就是这两个维度的耦合）
* 显存特征保持不变（v1 显存端已经 RMSE 902 MiB，无改动必要）
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

# 复用 v1 的 MODEL_META / DATASET_TOKENS / memory_features / ridge_fit / CARD_MEMORY_MIB
import fit_rtx4090_vl_v1 as v1

REPO = Path(__file__).resolve().parents[1]
V1_DATA = REPO / "artifacts" / "rtx4090_vl_observations_20260823"
OOS_DATA = V1_DATA / "oos_v1_results.jsonl"
SUPP_DATA = V1_DATA / "supp_v11_results.jsonl"
DEFAULT_OUT = REPO / "artifacts" / "rtx4090_vl_v11_predictor.json"


def throughput_features(cfg: dict):
    """v1.1 吞吐特征：v1 的 8 个 + `gc × log_gpu` 交叉项。"""
    m = v1.MODEL_META[cfg["model"]]
    param_scale = m["params_b"] / 4.0
    tps = cfg.get("tokens_per_sample")
    if tps is None:
        tps = v1.DATASET_TOKENS.get(cfg["dataset"])
    tokens_per_sample = float(tps)
    mbs = float(cfg["mbs"])
    gc = 1.0 if cfg["gc"] else 0.0
    zero = cfg["zero"]
    gpu = float(cfg["gpu_count"])
    is_z2 = 1.0 if zero == "z2" else 0.0
    is_z3 = 1.0 if zero == "z3" else 0.0

    feats = [
        1.0,
        np.log2(mbs),
        np.log2(gpu),
        -gc,
        -np.log2(1.0 + tokens_per_sample / 1000.0),
        -np.log2(param_scale + 1),
        -is_z3 * np.log2(gpu),
        is_z2 * np.log2(gpu),
        gc * np.log2(gpu),   # 新增：gc 与卡数的交叉项
    ]
    names = [
        "const", "log_mbs", "log_gpu", "neg_gc",
        "neg_log1p_tokens_kt", "neg_log_param_scale",
        "z3_pcie_penalty", "z2_scaling",
        "gc_x_log_gpu",   # 新增
    ]
    return np.array(feats, dtype=float), names


def load_all():
    rows = []
    # v1 原始
    for line in open(V1_DATA / "matrix_results_fixed.jsonl"):
        r = json.loads(line); r["source"] = "v1_matrix"; rows.append(r)
    # OOS（剔除框架失败：peak<1000 且非 OOM）
    if OOS_DATA.is_file():
        for line in open(OOS_DATA):
            r = json.loads(line)
            if r["exit"] != 0 and not r["oom"]:
                continue  # 框架失败（如单卡 z2/z3）
            r["source"] = "oos_v1"; rows.append(r)
    # 补数
    if SUPP_DATA.is_file():
        for line in open(SUPP_DATA):
            r = json.loads(line)
            if r["exit"] != 0 and not r["oom"]:
                continue
            r["source"] = "supp_v11"; rows.append(r)
    return rows


def fit_throughput_v11(rows):
    ok = [r for r in rows if r.get("track") != "text_smoke"
          and r["exit"] == 0 and not r["oom"]
          and r.get("tokens_per_s") and r["tokens_per_s"] > 0]
    X, names = [], None
    y = []
    for r in ok:
        f, n = throughput_features(r)
        X.append(f); y.append(np.log(float(r["tokens_per_s"])))
        if names is None: names = n
    X = np.asarray(X); y = np.asarray(y)
    w, alpha = v1.ridge_fit(X, y, alphas=[0.01, 0.1, 1, 3, 10, 30])
    pred = X @ w; resid = y - pred
    return dict(
        coefficients=dict(zip(names, w.tolist())),
        feature_names=names,
        alpha=alpha,
        loo_mse=float(np.mean(resid ** 2)),
        loo_rmse_log=float(np.sqrt(np.mean(resid ** 2))),
        approx_mape=float(np.mean(np.abs(np.exp(resid) - 1))),
        n_train=len(ok),
    )


def fit_memory_v11(rows):
    """显存特征沿用 v1；只把新数据加进来。"""
    vl = [r for r in rows if r.get("track") != "text_smoke"]
    ok = [r for r in vl if r["exit"] == 0 and not r["oom"] and r.get("peak_mib")]
    X, names = [], None; y = []
    for r in ok:
        f, n = v1.memory_features(r)
        X.append(f); y.append(float(r["peak_mib"]))
        if names is None: names = n
    X = np.asarray(X); y = np.asarray(y)
    w, alpha = v1.ridge_fit(X, y, alphas=[0.1, 1, 3, 10, 30, 100, 300])
    pred = X @ w; resid = y - pred
    safety = float(max(np.quantile(np.maximum(resid, 0), 0.95), 0.0))

    oom_rows = [r for r in vl if r["oom"]]
    tp, fn = 0, 0
    for r in oom_rows:
        f, _ = v1.memory_features(r)
        upper = float(f @ w) + safety
        if upper + 800 >= v1.CARD_MEMORY_MIB: tp += 1
        else: fn += 1
    fp = 0
    for r in ok:
        f, _ = v1.memory_features(r)
        upper = float(f @ w) + safety
        if upper + 800 >= v1.CARD_MEMORY_MIB and r["peak_mib"] < 22000: fp += 1
    return dict(
        coefficients=dict(zip(names, w.tolist())),
        feature_names=names,
        alpha=alpha,
        loo_mse=float(np.mean((y - pred) ** 2)),
        loo_rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
        loo_mae=float(np.mean(np.abs(y - pred))),
        safety_margin_mib=safety,
        n_train=len(ok),
        oom_admission=dict(tp=tp, fn=fn, total_oom=len(oom_rows)),
        false_reject_on_ok=fp,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load_all()
    from collections import Counter
    src = Counter(r.get("source", "?") for r in rows)
    print(f"载入 {len(rows)} 行  来源分布: {dict(src)}")

    mem = fit_memory_v11(rows)
    print(f"\n显存: alpha={mem['alpha']}  n={mem['n_train']}  "
          f"RMSE={mem['loo_rmse']:.0f}  MAE={mem['loo_mae']:.0f}  "
          f"safety={mem['safety_margin_mib']:.0f}")
    print(f"  OOM 召回: {mem['oom_admission']}")
    print(f"  ok 错拒: {mem['false_reject_on_ok']}")

    tp = fit_throughput_v11(rows)
    print(f"\n吞吐: alpha={tp['alpha']}  n={tp['n_train']}  "
          f"log-RMSE={tp['loo_rmse_log']:.3f}  MAPE≈{tp['approx_mape']:.1%}")
    print("  系数:")
    for k, v in tp["coefficients"].items():
        print(f"    {k:22} {v:+.3f}")

    artifact = dict(
        name="rtx4090_vl_v11",
        version="1.1.0",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        card_memory_mib=v1.CARD_MEMORY_MIB,
        model_meta=v1.MODEL_META,
        dataset_tokens=v1.DATASET_TOKENS,
        n_observations=len(rows),
        source_breakdown=dict(src),
        memory_model=mem,
        throughput_model=tp,
        notes="v1.1：合并 v1+OOS+补数，吞吐加 gc×log_gpu 交叉项",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"\n冻结产物写入 {args.out}")


if __name__ == "__main__":
    main()
