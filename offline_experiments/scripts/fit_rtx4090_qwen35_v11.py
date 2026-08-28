#!/usr/bin/env python3
"""4090 Qwen3.5 独立预测器 v1 的拟合脚本。

设计原则（与 4090 VL 预测器同一套方法论，但特征完全重设计）
------------------------------------------------------------
* **独立**：不与 H800/A100 混合，也不与 4090 VL 混合。Qwen3.5 是混合注意力
  （1/4 层 full + 3/4 层 linear），显存与吞吐机制和稠密 VL 模型不同。
* **核心结构：cutoff 的角色反转**。这是 Qwen3.5 在 4090 上最重要的机制：
      packing 关 → 每样本有效 token = 数据集自然长度，cutoff 完全不影响
      packing 开 → 每样本有效 token ≈ 1.27 × cutoff，与数据集无关
  实测证据：short_512 在 c2048/c4096/c8192 三档下 packing 关时都是 311
  tok/sample（变异 0%）；packing 开时变成 2650/5210/10330（变异 0%）。
* **FA2 无 packing 的吞吐塌陷**：4090 无 FA3，只能 FA2；FA2 触发 llamafactory
  的 Qwen3.5 patcher 把 position_ids 转 cu_seqlens，此时 fla 强制 batch=1，
  吞吐塌到 1/6～1/26。这条用 `fa2_nopack` 特征显式建模。
* **sdpa + packing 必失败**：注意力掩码尺寸不匹配，实测 12 组全挂。
  拟合时剔除，预测器里做前置门直接拒绝。

模型形式
--------
    peak_mib   = intercept + Σ w_i * f_i(config)
    peak_upper = peak_mib  + safety_margin      （残差 95 分位）
    log(tps)   = intercept + Σ v_i * g_i(config)
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
DATA_DIR = REPO / "artifacts" / "rtx4090_qwen35_observations_20260825"
DEFAULT_OUT = REPO / "artifacts" / "rtx4090_qwen35_v11_predictor.json"

DATA_FILES = [
    "qwen35_fa2_matrix.jsonl",
    "qwen35_sdpa_matrix.jsonl",
    "qwen35_packing_single.jsonl",
    "qwen35_packing_multi.jsonl",
    "oos_v1_results.jsonl",   # v1.1 新增：20 组 out-of-sample（14 有效 + 6 框架失败剔除）
]

# 三个模型的几何。混合注意力：full_attention_interval=4，即 1/4 层是 full。
# 三个模型的 full 层占比都是 0.25，所以该比例是常数、不入模。
MODEL_META = {
    "qwen35_0p8b": dict(params_b=0.87, hidden=1024, layers=24,
                        full_layers=6, kv_heads=2, head_dim=256,
                        intermediate=3584),
    "qwen35_4b":   dict(params_b=4.66, hidden=2560, layers=32,
                        full_layers=8, kv_heads=4, head_dim=256,
                        intermediate=9216),
    "qwen35_9b":   dict(params_b=9.65, hidden=4096, layers=32,
                        full_layers=8, kv_heads=4, head_dim=256,
                        intermediate=12288),
}

# 数据集的 token 长度分布（用 Qwen3.5 tokenizer 实测 1000 条样本得到）。
#
# 关键区分：**显存看尾部，吞吐看均值**。
#   峰值显存由批次里最长的样本决定 —— 用 max（再被 cutoff 截断）
#   吞吐是总 token / 总时间 —— 用 mean
# 这两个数在 multiturn_4096 上差 3.6 倍，混用会让显存严重低估。
DATASET_LEN = {
    # dataset: (mean, max)  —— mean 给吞吐，max 给显存
    "short_512":      (153.0,  445.0),
    "multiturn_4096": (1159.0, 4201.0),
    # blind_f1 / blind_f2 是图像轨道，每样本图数固定、长度均匀
    # （由 tokens_seen 反解的变异只有 0.5%），mean ≈ max
    "blind_f1":       (811.0,  811.0),
    "blind_f2":       (1538.0, 1538.0),
}

# packing 开启时每样本有效 token / cutoff 的实测比例
# 实测 c2048=1.294  c4096=1.272  c8192=1.261，随 cutoff 缓慢下降，取均值
PACK_TOKEN_RATIO = 1.276

CARD_MEMORY_MIB = 24564


def normalize_row(r: dict) -> dict:
    """两套 schema 统一：矩阵文件用 cutoff_len+ga，packing 文件用 cutoff+pack_mode。"""
    c = dict(r)
    c["cutoff"] = r.get("cutoff") if r.get("cutoff") is not None else r.get("cutoff_len")
    c["ga"] = r.get("ga") if r.get("ga") is not None else 1
    # 矩阵文件没有 pack_mode 字段，那些作业全部 packing=False
    c["pack_mode"] = r.get("pack_mode") or "off"
    return c


def _dataset_len(cfg: dict) -> tuple[float, float]:
    """返回该数据集的 (mean, max) token 长度。允许显式覆盖。"""
    if cfg.get("tokens_per_sample") is not None:
        t = float(cfg["tokens_per_sample"])
        return t, t
    pair = DATASET_LEN.get(cfg["dataset"])
    if pair is None:
        raise ValueError(f"未知数据集 {cfg.get('dataset')}，请显式给 tokens_per_sample")
    return pair


def memory_tokens(cfg: dict) -> float:
    """驱动**峰值显存**的每样本 token 数。

    这是 cutoff 角色反转的落点：
      packing 开 → 变长序列被拼满到 cutoff，有效长度 ≈ 1.276 × cutoff
      packing 关 → 由批次里**最长**的样本决定，并被 cutoff 截断：
                   min(cutoff, 数据集最长样本)

    为什么用 max 而不是 mean：峰值显存是整个训练过程中出现过的最高水位，
    由最长的那个批次决定。multiturn_4096 的 max/mean = 3.6，用 mean 会让
    显存低估 2.8 倍（实测：26 个 OOM 配置被误判为可放行）。
    """
    pm = cfg.get("pack_mode", "off")
    cutoff = float(cfg["cutoff"])
    if pm in ("neat", "pack"):
        return PACK_TOKEN_RATIO * cutoff
    _, dmax = _dataset_len(cfg)
    return min(cutoff, dmax)


def throughput_tokens(cfg: dict) -> float:
    """驱动**吞吐**的每样本 token 数——用均值，不是尾部。

    吞吐 = 总 token / 总时间，是整个数据集的平均行为，跟最长样本无关。
    """
    pm = cfg.get("pack_mode", "off")
    if pm in ("neat", "pack"):
        return PACK_TOKEN_RATIO * float(cfg["cutoff"])
    dmean, _ = _dataset_len(cfg)
    return min(float(cfg["cutoff"]), dmean)


def memory_features(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """显存特征。自由变量：model, mbs, ga, gc, zero, gpu_count, attn,
    pack_mode, cutoff, dataset。"""
    m = MODEL_META[cfg["model"]]
    param_gib = m["params_b"] * 2 / 1.024**3   # BF16 权重
    hidden_scale = m["hidden"] / 1024.0        # 1.0 / 2.5 / 4.0，三值不共线
    mbs = float(cfg["mbs"])
    ga = float(cfg.get("ga", 1))
    gc = 1.0 if cfg["gc"] else 0.0
    zero = cfg["zero"]
    gpu = float(cfg["gpu_count"])
    is_z2 = 1.0 if zero == "z2" else 0.0
    is_z3 = 1.0 if zero == "z3" else 0.0
    is_sdpa = 1.0 if cfg["attn"] == "sdpa" else 0.0
    is_pack = 1.0 if cfg.get("pack_mode", "off") in ("neat", "pack") else 0.0

    eff_tok = memory_tokens(cfg)
    weight_gib = param_gib / gpu if zero == "z3" else param_gib
    # FA2+packing 下 llamafactory 强制 batch=1，实际微批不是 mbs
    eff_mbs = 1.0 if (is_pack and not is_sdpa) else mbs
    activation = eff_mbs * eff_tok / 1000.0
    # GC 用**互斥的两组斜率**表达，不用加减法。
    #
    # 踩过的三个坑：
    #   1) activation + activation*(1-gc)：gc=False 占 91%，两项几乎共线，
    #      ridge 把系数摆成 ±3000 的对冲。
    #   2) activation + activation*gc：gc 项拟合成 -2210，线性外推到长序列
    #      大 mbs 时放大成 -36000 MiB，预测出负显存（-11407）。
    #   3) 只把基础项按 gc 拆开、hidden 交叉项仍共享：hidden 交叉项由 91% 的
    #      gc=False 行决定（+699），逼得 gc 基础项做负补偿（-1543）。
    #
    # 正确做法：把「基础斜率」和「hidden 交叉」**成对**按 gc 拆成两组互斥特征。
    # GC 改变的是每单位激活的显存斜率本身，而这个斜率又随 hidden 变化，
    # 所以两者都得分开估。
    # 实测支持：gc=True 子集内 peak 对激活的单变量斜率是
    #   0.8B +184 MiB/单位（r=0.975）、4B +387 MiB/单位（r=0.979），
    # 都是正的，且随 hidden 增大——正好对应「基础 + hidden 交叉」两个正系数。
    activation_nogc = activation * (1.0 - gc)
    activation_gc = activation * gc
    optim_share = 1.0 / gpu if (is_z2 or is_z3) else 1.0

    feats = [
        1.0,                              # 框架常驻开销
        weight_gib,                       # 权重
        hidden_scale,                     # 与 hidden 成比例的常驻项（norm/embed 等）
        eff_mbs,                          # base activation & KV cache
        activation_nogc,                  # 关 GC：每单位激活的基础显存
        hidden_scale * activation_nogc,   # 关 GC：hidden 放大部分
        activation_gc,                    # 开 GC：每单位激活的基础显存
        hidden_scale * activation_gc,     # 开 GC：hidden 放大部分
        gc * np.log2(gpu),                # 开 GC 时随卡数上升的显存加成
                                          #（v1.0 全部 gc 行是单卡，log2(1)=0，
                                          #  该项恒零无法识别；v1.1 补了多卡 GC）
        is_sdpa * activation,             # sdpa 需要显式物化注意力矩阵
        is_pack,                          # packing 自身的缓冲开销
        is_z2 * optim_share,              # z2 优化器分片
        is_z3 * optim_share,              # z3 优化器 + 权重双分片
        is_z3,                            # z3 all-gather 峰值加成
        is_z3 * eff_mbs,                  # z3 下微批对 all-gather 缓冲的压力
        is_z3 * np.log2(gpu),             # z3 all-gather 瞬时峰值随卡数上升
        np.log2(ga),                      # 梯度累积（实测影响小，让 ridge 定）
    ]
    names = [
        "const", "weight_gib", "hidden_scale", "eff_mbs",
        "activation_nogc", "hidden_x_activation_nogc",
        "activation_gc", "hidden_x_activation_gc",
        "gc_x_log_gpu",
        "sdpa_x_activation", "is_pack",
        "z2_optim_shard", "z3_optim_shard", "z3_bias", "z3_eff_mbs",
        "z3_x_log_gpu", "log_ga",
    ]
    return np.array(feats, dtype=float), names


def throughput_features(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """吞吐特征（log 空间）。

    三个 regime 决定吞吐的量级（中位 tok/s 实测）：
        fa2 无 packing   1027   ← 基准。FA2 触发 llamafactory 的 Qwen3.5
                                  patcher，fla 强制 batch=1，吞吐塌陷
        sdpa 无 packing  1821   ← 不触发 patcher，mbs 可用
        fa2 + packing    9887   ← 保留 FA2 又把变长序列拼满，最优
    用 `is_sdpa` 和 `is_pack` 两个哑变量相对基准编码，不再单独放
    `fa2_nopack` 项（那样三者与常数项共线）。

    两个来自实测的结构约束：
      1. `log_mbs` 只在 sdpa 下起作用。fa2 被逼成 batch=1，实验里 mbs 从未
         变化过（可比对配对数 = 0），强行放全局 log_mbs 只会串到别的系数上。
      2. 不放 `z2_scaling`。单卡必然 zero=none、多卡必然 z2/z3，
         `is_z2 × log_gpu` 与 `log_gpu` 几乎重合；第一次拟合里它把 log_gpu
         的系数吸走了（+0.033，而实测卡数扩展效率是 0.95~0.99）。
         正确的参数化是：log_gpu 承担主扩展，z3 只作为偏离项。
      3. 不放 `gc × log_gpu`（VL v1.1 里有效的那一项）。本数据集 48 条
         `gc=True` 行**全部是单卡**，log2(1)=0，该特征恒为零、无法识别。
         等有了多卡 gc 数据再加。

    试过但无效、已撤回的两项（记下来免得重复试）：
      `is_single_card` 与 `sdpa × (log2 mbs)²`。动机是残差按 gpu 分组时单卡
      偏高 8.9%、按 mbs 分组时 mbs=4 偏高 11.9%，看起来像缺了单卡固定开销和
      MBS 饱和项。加上后两个系数都是 +0.03 量级、LOO MAPE 从 19.0% 微升到
      19.2%。结论：那些是**边际**残差模式，在共变量不平衡的设计里不等于缺
      特征，不能照着边际残差加项。
    """
    m = MODEL_META[cfg["model"]]
    param_scale = m["params_b"] / 4.0
    mbs = float(cfg["mbs"])
    ga = float(cfg.get("ga", 1))
    gc = 1.0 if cfg["gc"] else 0.0
    zero = cfg["zero"]
    gpu = float(cfg["gpu_count"])
    is_z3 = 1.0 if zero == "z3" else 0.0
    is_sdpa = 1.0 if cfg["attn"] == "sdpa" else 0.0
    is_pack = 1.0 if cfg.get("pack_mode", "off") in ("neat", "pack") else 0.0
    is_neat = 1.0 if cfg.get("pack_mode") == "neat" else 0.0
    eff_tok = throughput_tokens(cfg)

    feats = [
        1.0,                                    # log 基准吞吐（fa2 无 packing）
        np.log2(gpu),                           # 卡数扩展（实测近线性，应 ≈1）
        -is_z3 * np.log2(gpu),                  # ZeRO-3 在 PCIe 上的扩展惩罚
        is_sdpa * np.log2(mbs),                 # MBS 只在 sdpa 下可用
        np.log2(ga),                            # 梯度累积减少同步点
        -gc,                                    # GC 损失
        gc * np.log2(gpu),                      # GC × 卡数交叉（v1.0 无法识别因为
                                                # 全部 gc 行是单卡；v1.1 有了多卡
                                                # GC 数据。VL v1.1 正是这项有效）
        np.log2(eff_tok / 1000.0),              # 每样本 token 越多，单步越划算
        -np.log2(param_scale + 1),              # 参数量惩罚
        is_sdpa,                                # sdpa regime 偏移
        is_pack,                                # packing regime 偏移（主增益）
        is_neat,                                # neat vs 普通 packing 的差异
    ]
    names = [
        "const", "log_gpu", "z3_pcie_penalty", "sdpa_x_log_mbs",
        "log_ga", "neg_gc", "gc_x_log_gpu",
        "log_eff_tokens_kt", "neg_log_param_scale",
        "is_sdpa", "is_pack", "is_neat",
    ]
    return np.array(feats, dtype=float), names


def ridge_fit(X: np.ndarray, y: np.ndarray, alphas: list[float]) -> tuple[np.ndarray, float]:
    """带 alpha 网格搜索的 ridge，用 leave-one-out 选 alpha。"""
    n, d = X.shape
    best_alpha, best_mse = None, np.inf
    for a in alphas:
        errs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            w = np.linalg.solve(X[mask].T @ X[mask] + a * np.eye(d),
                                X[mask].T @ y[mask])
            errs.append((X[i] @ w - y[i]) ** 2)
        mse = float(np.mean(errs))
        if mse < best_mse:
            best_mse, best_alpha = mse, a
    w = np.linalg.solve(X.T @ X + best_alpha * np.eye(d), X.T @ y)
    return w, best_alpha


def loo_predictions(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """每行的留一预测，用于报告诚实的泛化误差（而非 in-sample 残差）。"""
    n, d = X.shape
    out = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        w = np.linalg.solve(X[mask].T @ X[mask] + alpha * np.eye(d),
                            X[mask].T @ y[mask])
        out[i] = X[i] @ w
    return out


def load_observations() -> list[dict]:
    rows = []
    for fn in DATA_FILES:
        for line in open(DATA_DIR / fn):
            r = normalize_row(json.loads(line))
            r["_file"] = fn
            rows.append(r)
    return rows


def is_infeasible(cfg: dict) -> str | None:
    """配置层面就跑不起来的组合，返回原因；可行返回 None。

    这三条都不是"显存装不下"，是训练根本起不来，所以必须在预测之前拦掉——
    否则预测器会给出一个显存数字并放行，等于推荐一个秒崩的配置。
    """
    pm = cfg.get("pack_mode", "off")
    if cfg["attn"] == "sdpa" and pm in ("neat", "pack"):
        return "sdpa + packing 必失败（注意力掩码尺寸不匹配），实测 12 组全挂"
    if cfg["gpu_count"] == 1 and cfg["zero"] in ("z2", "z3"):
        return "DeepSpeed 不支持单卡 zero-2/3"
    if cfg["attn"] == "fa2" and float(cfg.get("mbs", 1)) > 1:
        # 2026-08-28 out-of-sample 实测发现（C 组 4 组 + D 组 2 组全部失败）：
        #   ValueError: The batch size is expected to be 1 rather than 2 when
        #   using `cu_seqlens`. Please flatten variable-length inputs ...
        #   fla/ops/gated_delta_rule/chunk.py:548 ← llamafactory patcher.py:208
        # FA2 触发 patcher 把 position_ids 转 cu_seqlens，fla 就要求 batch=1，
        # 与开不开 packing 无关。这解释了为什么拟合池里 fa2 的 292 行全是 mbs=1。
        # 推荐含义：4090 上要 MBS>1 只能走 sdpa（放弃 packing）；
        #           要 packing 只能 fa2 + mbs=1。
        return ("FA2 路径下 mbs 必须为 1（patcher 转 cu_seqlens 后 fla 强制 "
                "batch=1，与 packing 无关）；要用更大 MBS 必须换 sdpa")
    return None


def fit_memory(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["exit"] == 0 and not r["oom"] and r.get("peak_mib")
          and not is_infeasible(r)]
    X, names, y = [], None, []
    for r in ok:
        f, n = memory_features(r)
        X.append(f); y.append(float(r["peak_mib"]))
        if names is None: names = n
    X = np.asarray(X); y = np.asarray(y)
    w, alpha = ridge_fit(X, y, alphas=[0.1, 1, 3, 10, 30, 100, 300, 1000])
    # 诚实的 LOO 误差
    loo_pred = loo_predictions(X, y, alpha)
    loo_resid = y - loo_pred
    # 安全边际用 in-sample 残差（上界要覆盖拟合点本身）
    resid = y - X @ w
    safety = float(max(np.quantile(np.maximum(resid, 0), 0.95), 0.0))

    oom_rows = [r for r in rows if r["oom"] and not is_infeasible(r)]
    tp = fn_ = 0
    for r in oom_rows:
        f, _ = memory_features(r)
        upper = float(f @ w) + safety
        if upper + 800 >= CARD_MEMORY_MIB: tp += 1
        else: fn_ += 1
    fp = 0
    for r in ok:
        f, _ = memory_features(r)
        upper = float(f @ w) + safety
        if upper + 800 >= CARD_MEMORY_MIB and r["peak_mib"] < 22000: fp += 1
    return dict(
        coefficients=dict(zip(names, w.tolist())),
        feature_names=names, alpha=alpha,
        loo_rmse=float(np.sqrt(np.mean(loo_resid ** 2))),
        loo_mae=float(np.mean(np.abs(loo_resid))),
        insample_rmse=float(np.sqrt(np.mean(resid ** 2))),
        safety_margin_mib=safety, n_train=len(ok),
        oom_admission=dict(tp=tp, fn=fn_, total_oom=len(oom_rows)),
        false_reject_on_ok=fp,
    )


def fit_throughput(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["exit"] == 0 and not r["oom"]
          and r.get("tokens_per_s") and r["tokens_per_s"] > 0
          and not is_infeasible(r)]
    X, names, y = [], None, []
    for r in ok:
        f, n = throughput_features(r)
        X.append(f); y.append(np.log(float(r["tokens_per_s"])))
        if names is None: names = n
    X = np.asarray(X); y = np.asarray(y)
    w, alpha = ridge_fit(X, y, alphas=[0.01, 0.1, 1, 3, 10, 30, 100])
    loo_pred = loo_predictions(X, y, alpha)
    loo_resid = y - loo_pred
    resid = y - X @ w
    return dict(
        coefficients=dict(zip(names, w.tolist())),
        feature_names=names, alpha=alpha,
        loo_rmse_log=float(np.sqrt(np.mean(loo_resid ** 2))),
        loo_mape=float(np.mean(np.abs(np.exp(loo_resid) - 1))),
        insample_mape=float(np.mean(np.abs(np.exp(resid) - 1))),
        n_train=len(ok),
    )


def dataset_sha(paths) -> str:
    h = sha256()
    for p in sorted(paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load_observations()
    infeas = [r for r in rows if is_infeasible(r)]
    print(f"载入 {len(rows)} 行观测")
    print(f"  其中配置层面不可行（剔除）：{len(infeas)} 行")
    for reason, cnt in Counter(is_infeasible(r) for r in infeas).items():
        print(f"    {cnt} 行：{reason}")

    mem = fit_memory(rows)
    print(f"\n显存模型: alpha={mem['alpha']}  n={mem['n_train']}")
    print(f"  LOO RMSE={mem['loo_rmse']:.0f} MiB  LOO MAE={mem['loo_mae']:.0f} MiB")
    print(f"  in-sample RMSE={mem['insample_rmse']:.0f} MiB")
    print(f"  安全边际={mem['safety_margin_mib']:.0f} MiB")
    print(f"  OOM 召回={mem['oom_admission']['tp']}/{mem['oom_admission']['total_oom']}"
          f" = {mem['oom_admission']['tp']/max(mem['oom_admission']['total_oom'],1)*100:.1f}%")
    print(f"  ok 行误拒={mem['false_reject_on_ok']}/{mem['n_train']}")
    print("  系数:")
    for k, v in mem["coefficients"].items():
        print(f"    {k:22} {v:+10.1f}")

    tp = fit_throughput(rows)
    print(f"\n吞吐模型: alpha={tp['alpha']}  n={tp['n_train']}")
    print(f"  LOO log-RMSE={tp['loo_rmse_log']:.3f}  LOO MAPE={tp['loo_mape']:.1%}")
    print(f"  in-sample MAPE={tp['insample_mape']:.1%}")
    print("  系数:")
    for k, v in tp["coefficients"].items():
        print(f"    {k:22} {v:+8.3f}")

    artifact = dict(
        name="rtx4090_qwen35_v1", version="1.0.0",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        card_memory_mib=CARD_MEMORY_MIB,
        model_meta=MODEL_META,
        dataset_len=DATASET_LEN,
        pack_token_ratio=PACK_TOKEN_RATIO,
        data_sha256=dataset_sha([DATA_DIR / f for f in DATA_FILES]),
        n_observations=len(rows),
        n_infeasible_excluded=len(infeas),
        memory_model=mem, throughput_model=tp,
        notes=(
            "独立发布的 4090 Qwen3.5 预测器；不与 H800/A100 或 4090 VL 混合。"
            "核心机制：packing 关时 cutoff 不影响显存，packing 开时 "
            "每样本有效 token ≈ 1.276 × cutoff。"
            "FA2 无 packing 会被 llamafactory patcher 逼成 batch=1，吞吐塌陷。"
            "sdpa + packing 与单卡 zero-2/3 均为配置层面不可行，预测器前置拒绝。"
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    print(f"\n冻结产物写入 {args.out}")


if __name__ == "__main__":
    main()
