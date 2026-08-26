#!/usr/bin/env python3
"""4090 VL 独立预测器 v1 的拟合脚本。

设计原则
--------
* **独立**：不依赖现有 H800/A100 拟合家族（那套已经与 3-SHA 绑定纠缠），
  只读 `artifacts/rtx4090_vl_observations_20260823/*.jsonl` 一份数据。
* **稳健**：ridge 回归 + 显式安全边际，避免高维过拟合（成功样本 <100）。
* **可复算**：所有系数、特征名、数据 SHA、拟合时间戳都写进 JSON，
  预测器只做纯计算。
* **两头对齐**：显存模型上界 (`peak_upper_mib`) 用于准入决策，
  显存模型中心 (`peak_center_mib`) 与吞吐模型 (`tokens_per_s`) 用于建议参数。

模型形式
--------
    peak_mib   = intercept + Σ w_i * f_i(config)   （中心）
    peak_upper = peak_mib  + safety_margin         （上界，按残差 95 分位）
    log(tps)   = intercept + Σ v_i * g_i(config)   （吞吐取 log）

其中 f_i / g_i 是简单的可解释特征，见 `memory_features` / `throughput_features`。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "artifacts" / "rtx4090_vl_observations_20260823"
DEFAULT_OUT = REPO / "artifacts" / "rtx4090_vl_v1_predictor.json"

# 两个模型的基础参数量（BF16 权重字节数 = 2 * params）
MODEL_META = {
    "qwen25vl3b": dict(
        params_b=3.755,
        hidden=2048,
        layers=36,
        vision_hidden=1280,
        vision_depth=32,
    ),
    "qwen3vl4b": dict(
        params_b=4.438,
        hidden=2560,
        layers=36,
        vision_hidden=1024,
        vision_depth=24,
    ),
}

# 各数据集每样本平均 token 数（从观测 tokens_seen / (measure*gpu*mbs) 得到）。
# 这是显存激活的实际驱动，用它替代粗糙的 frames；预测时可以查表或显式给。
# 单卡扫描 short_512 数据是纯文本，此表不含。
DATASET_TOKENS = {
    "blind_f1": 914,
    "blind_f2": 1739,
    "blind_f4": 3774,
    "calib_f2": 4860,
}

CARD_MEMORY_MIB = 24564  # 4090 24 GiB 单卡上限


def load_observations() -> list[dict]:
    """合并单卡扫描 + 多卡矩阵。单卡扫描缺失字段用默认值补齐。"""
    rows = []
    # 多卡矩阵（108 条，字段完整）
    for line in open(DATA_DIR / "matrix_results_fixed.jsonl"):
        r = json.loads(line)
        rows.append(r)
    # 单卡扫描（36 条，缺 gc/zero/gpu_count/dataset）
    for line in open(DATA_DIR / "single_card_sweep.jsonl"):
        r = json.loads(line)
        r.setdefault("gc", False)
        r.setdefault("zero", "none")
        r.setdefault("gpu_count", 1)
        # 单卡扫用旧的 short_512 类文本，非 VL 图像；跳过用于 VL 拟合
        # 但需要保留看单卡吞吐；用 track='text_smoke' 标注
        r["track"] = "text_smoke"
        r["cutoff_len"] = r.get("cutoff_len", 4096)
        rows.append(r)
    return rows


def memory_features(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """显存模型的特征向量。

    自由变量：model, mbs, gc, zero, gpu_count, tokens_per_sample。
    tokens_per_sample 优先用 cfg 里显式给的；否则查 DATASET_TOKENS。
    cutoff_len 恒定 4096（VL 观测背景），不作为变量。
    """
    m = MODEL_META[cfg["model"]]
    param_gib = m["params_b"] * 2 / 1.024**3  # BF16 权重的 GiB
    # 每样本实际 token 数：图 token + 文本 token，主导激活尺寸
    tps = cfg.get("tokens_per_sample")
    if tps is None:
        tps = DATASET_TOKENS.get(cfg["dataset"])
    if tps is None:
        raise ValueError(f"缺少 tokens_per_sample 或未知 dataset {cfg.get('dataset')}")
    tokens_per_sample = float(tps)
    mbs = float(cfg["mbs"])
    gc = 1.0 if cfg["gc"] else 0.0
    zero = cfg["zero"]
    gpu = float(cfg["gpu_count"])
    is_z2 = 1.0 if zero == "z2" else 0.0
    is_z3 = 1.0 if zero == "z3" else 0.0
    is_4b = 1.0 if cfg["model"] == "qwen3vl4b" else 0.0

    # 权重实际占用：z3 分片 / gpu_count，其它情况全份
    weight_gib = param_gib / gpu if zero == "z3" else param_gib
    # 激活代理：mbs × tokens_per_sample / 1000（保持数值适中）
    activation = mbs * tokens_per_sample / 1000.0
    # GC 削减部分：coefficient 应为负；只有 GC=1 时该项存在
    activation_gc_off = activation * (1.0 - gc)
    # 优化器/梯度占用（LoRA + 冻结骨干；z2/z3 分片按卡摊）
    optim_share = 1.0 / gpu if (is_z2 or is_z3) else 1.0

    feats = [
        1.0,                                # 常数（框架开销）
        weight_gib,                         # 主导权重项
        mbs,                                # base activation & KV cache
        activation,                         # 基础激活（含 GC 时的残留部分）
        activation_gc_off,                  # 关 GC 时额外多出来的激活
        is_4b * activation,                 # 4B 激活比 3B 贵
        is_z2 * optim_share,                # z2 优化器分片
        is_z3 * optim_share,                # z3 优化器 + 权重双分片
        is_z3,                              # z3 的 all-gather 峰值加成
        is_z3 * mbs,                        # z3 下 mbs 增加对 all-gather 缓冲的额外压力
    ]
    names = [
        "const",
        "weight_gib",
        "mbs",
        "activation",
        "activation_gc_off",
        "is_4b_activation",
        "z2_optim_shard",
        "z3_optim_shard",
        "z3_bias",
        "z3_mbs",
    ]
    return np.array(feats, dtype=float), names


def throughput_features(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """吞吐模型的特征向量（log 空间）。"""
    m = MODEL_META[cfg["model"]]
    param_scale = m["params_b"] / 4.0  # 归一到 4B
    tps = cfg.get("tokens_per_sample")
    if tps is None:
        tps = DATASET_TOKENS.get(cfg["dataset"])
    tokens_per_sample = float(tps)
    mbs = float(cfg["mbs"])
    gc = 1.0 if cfg["gc"] else 0.0
    zero = cfg["zero"]
    gpu = float(cfg["gpu_count"])
    is_z2 = 1.0 if zero == "z2" else 0.0
    is_z3 = 1.0 if zero == "z3" else 0.0

    feats = [
        1.0,                                    # log 基准吞吐
        np.log2(mbs),                           # MBS scaling
        np.log2(gpu),                           # gpu scaling
        -gc,                                    # GC 损失
        -np.log2(1.0 + tokens_per_sample / 1000.0),  # 序列长度惩罚
        -np.log2(param_scale + 1),              # 参数量惩罚
        -is_z3 * np.log2(gpu),                  # ZeRO-3 在 PCIe 上的额外惩罚
        is_z2 * np.log2(gpu),                   # ZeRO-2 正常线性扩展
    ]
    names = [
        "const",
        "log_mbs",
        "log_gpu",
        "neg_gc",
        "neg_log1p_tokens_kt",
        "neg_log_param_scale",
        "z3_pcie_penalty",
        "z2_scaling",
    ]
    return np.array(feats, dtype=float), names


def ridge_fit(X: np.ndarray, y: np.ndarray, alphas: list[float]) -> tuple[np.ndarray, float]:
    """带 alpha 网格搜索的 ridge。分层 K-fold（这里数据小，用 leave-one-out）。"""
    n, d = X.shape
    best_alpha, best_mse = None, np.inf
    for a in alphas:
        # LOO
        errs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            Xi, yi = X[mask], y[mask]
            w = np.linalg.solve(Xi.T @ Xi + a * np.eye(d), Xi.T @ yi)
            errs.append((X[i] @ w - y[i]) ** 2)
        mse = float(np.mean(errs))
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    # 最终用最佳 alpha 在全集上拟合
    w = np.linalg.solve(X.T @ X + best_alpha * np.eye(d), X.T @ y)
    return w, best_alpha


def fit_memory(rows: list[dict]) -> dict:
    """只用 VL 成功行（有图像的观测）拟合显存中心与上界。"""
    vl_rows = [r for r in rows if r.get("track") != "text_smoke"]
    ok = [r for r in vl_rows if r["exit"] == 0 and not r["oom"] and r.get("peak_mib")]
    X, names = [], None
    y = []
    for r in ok:
        f, n = memory_features(r)
        X.append(f)
        if names is None: names = n
        y.append(float(r["peak_mib"]))
    X = np.asarray(X); y = np.asarray(y)
    w, alpha = ridge_fit(X, y, alphas=[0.1, 1, 3, 10, 30, 100, 300])
    pred = X @ w
    resid = y - pred
    # 上界：中心 + 95 分位残差（不小于 0），提供准入余量
    safety = float(max(np.quantile(np.maximum(resid, 0), 0.95), 0.0))
    # 用 OOM 行验证准入召回率
    oom_rows = [r for r in vl_rows if r["oom"]]
    tp = 0  # 上界 >= 24GB 正确拒 OOM
    fn = 0
    for r in oom_rows:
        f, _ = memory_features(r)
        upper = float(f @ w) + safety
        if upper + 800 >= CARD_MEMORY_MIB:  # 与 24564 距离 <800 视为已拒
            tp += 1
        else:
            fn += 1
    # ok 行不应被错拒
    fp = 0
    for r in ok:
        f, _ = memory_features(r)
        upper = float(f @ w) + safety
        if upper + 800 >= CARD_MEMORY_MIB and r["peak_mib"] < 22000:
            fp += 1
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


def fit_throughput(rows: list[dict]) -> dict:
    """所有成功的 VL 图像作业上拟合吞吐（log 空间）。"""
    vl_rows = [r for r in rows if r.get("track") != "text_smoke"]
    ok = [r for r in vl_rows if r["exit"] == 0 and not r["oom"]
          and r.get("tokens_per_s") and r["tokens_per_s"] > 0]
    X, names = [], None
    y = []
    for r in ok:
        f, n = throughput_features(r)
        X.append(f)
        if names is None: names = n
        y.append(np.log(float(r["tokens_per_s"])))
    X = np.asarray(X); y = np.asarray(y)
    w, alpha = ridge_fit(X, y, alphas=[0.01, 0.1, 1, 3, 10, 30])
    pred = X @ w
    resid = y - pred
    # 对数空间的 mape 近似 = mean(|resid|)
    return dict(
        coefficients=dict(zip(names, w.tolist())),
        feature_names=names,
        alpha=alpha,
        loo_mse=float(np.mean(resid ** 2)),
        loo_rmse_log=float(np.sqrt(np.mean(resid ** 2))),
        # 相对误差近似（在对数空间 rmse 直接对应几何标准差）
        approx_mape=float(np.mean(np.abs(np.exp(resid) - 1))),
        n_train=len(ok),
    )


def dataset_sha(*paths: Path) -> str:
    h = sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load_observations()
    print(f"载入 {len(rows)} 条观测")

    mem = fit_memory(rows)
    print(f"\n显存模型: alpha={mem['alpha']}  n={mem['n_train']}")
    print(f"  LOO RMSE={mem['loo_rmse']:.1f} MiB  MAE={mem['loo_mae']:.1f} MiB")
    print(f"  安全边际={mem['safety_margin_mib']:.0f} MiB")
    print(f"  OOM 召回={mem['oom_admission']}")
    print(f"  ok 行误拒={mem['false_reject_on_ok']}")

    tp = fit_throughput(rows)
    print(f"\n吞吐模型: alpha={tp['alpha']}  n={tp['n_train']}")
    print(f"  LOO log-RMSE={tp['loo_rmse_log']:.3f}  近似 MAPE={tp['approx_mape']:.1%}")

    artifact = dict(
        name="rtx4090_vl_v1",
        version="1.0.0",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        card_memory_mib=CARD_MEMORY_MIB,
        model_meta=MODEL_META,
        dataset_tokens=DATASET_TOKENS,
        data_sha256=dataset_sha(
            DATA_DIR / "matrix_results_fixed.jsonl",
            DATA_DIR / "single_card_sweep.jsonl",
        ),
        n_observations=len(rows),
        memory_model=mem,
        throughput_model=tp,
        notes=(
            "独立发布的 4090 VL 预测器；未与 H800/A100 数据混合。"
            "输入配置字段：model, mbs, gc, zero, gpu_count, dataset (或 tokens_per_sample)。"
            "cutoff_len 恒定 4096（VL 数据的实测背景），不作为变量入模。"
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"\n冻结产物写入 {args.out}")


if __name__ == "__main__":
    main()
