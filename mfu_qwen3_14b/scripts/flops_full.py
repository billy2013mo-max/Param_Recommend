#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3 全参(FULL) SFT 逻辑模型 FLOPs 计算器。

与 8B LoRA 版（mfu_qwen3_8b/scripts/flops_lora.py）的唯一区别是可训练性口径：
  - 全参：每个 Linear 都要算 dW → 6*M*K*N（forward 2 + dX 2 + dW 2）
  - lm_head 也可训练 → 同样 6*M*K*N（LoRA 版是冻结的 4MKN）
  - attention core：12 * L * hidden * ΣS²  —— 与 LoRA 版逐字相同
  - activation recomputation 不进分子（原则 §3.1）

自校验：与《MFU 实验统计原则》的分量拆法对齐，并与 8B LoRA 版共用 attention 公式；
另外用 2 卡 baseline 报告的 84.85 GFLOP/token（不含 attention core 口径）交叉核对。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "ModelShape":
        hidden = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        return cls(
            num_layers=cfg["num_hidden_layers"],
            hidden=hidden,
            num_heads=heads,
            num_kv_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim", hidden // heads),
            intermediate=cfg["intermediate_size"],
            vocab=cfg["vocab_size"],
        )


def linear_dims(ms: ModelShape) -> dict[str, tuple[int, int]]:
    """每个 Linear 的 (输入维, 输出维)。

    注意 q/o_proj 用 num_heads*head_dim 而不是 hidden——Qwen3 两者相等时无差别，
    但 head_dim != hidden/num_heads 的模型上假设 hidden×hidden 会算错（历史 bug）。
    """
    q_out = ms.num_heads * ms.head_dim
    kv_out = ms.num_kv_heads * ms.head_dim
    return {
        "q_proj": (ms.hidden, q_out),
        "k_proj": (ms.hidden, kv_out),
        "v_proj": (ms.hidden, kv_out),
        "o_proj": (q_out, ms.hidden),
        "gate_proj": (ms.hidden, ms.intermediate),
        "up_proj": (ms.hidden, ms.intermediate),
        "down_proj": (ms.intermediate, ms.hidden),
    }


ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


def step_flops(ms: ModelShape, seq_lengths: list[int]) -> dict[str, int]:
    """一个 optimizer step 的逻辑 FLOPs 分量（全参口径）。

    seq_lengths: 该 step 全部 microbatch 里每条序列的长度。
      Linear 项传 padded 长度（GEMM 真的算了 padding），
      attention core 传真实长度（FA 不算 padding 的 pair）。
    """
    dims = linear_dims(ms)
    tokens = sum(seq_lengths)
    L = ms.num_layers

    # 全参可训练 Linear：forward + dX + dW = 6*M*K*N
    attn_proj = 6 * tokens * L * sum(dims[n][0] * dims[n][1] for n in ATTN_PROJ)
    mlp = 6 * tokens * L * sum(dims[n][0] * dims[n][1] for n in MLP_PROJ)
    lm_head = 6 * tokens * ms.hidden * ms.vocab

    attn_core = 12 * L * ms.hidden * sum(s * s for s in seq_lengths)

    parts = {
        "attn_projection": attn_proj,
        "mlp": mlp,
        "lm_head": lm_head,
        "attention_core": attn_core,
    }
    parts["total"] = sum(parts.values())
    return parts


QWEN3_14B = ModelShape(
    num_layers=40, hidden=5120, num_heads=40, num_kv_heads=8,
    head_dim=128, intermediate=17408, vocab=151936,
)


def selfcheck() -> bool:
    """两项交叉核对，都必须过。"""
    ok = True

    # ① Linear 部分的每 token FLOPs 必须与手算一致
    p = step_flops(QWEN3_14B, [1])
    lin = p["attn_projection"] + p["mlp"] + p["lm_head"]
    L, H, NH, NKV, HD, I, V = 40, 5120, 40, 8, 128, 17408, 151936
    q, kv = NH * HD, NKV * HD
    hand = 6 * L * ((H * q) + 2 * (H * kv) + (q * H) + 2 * (H * I) + (I * H)) + 6 * H * V
    print(f"① Linear/token: 公式 {lin} vs 手算 {hand} → "
          f"{'一致' if lin == hand else '不一致'}")
    ok &= lin == hand

    # ② 与同事 2 卡 baseline 报告的 84.85 GFLOP/token 对比。
    #    该口径不含 attention core（报告里 attn核仅占 1.1%，单列），故只比 Linear 部分。
    got = lin / 1e9
    print(f"② Linear/token = {got:.2f} GFLOP  vs 2卡报告 84.85 GFLOP（不含 attn 核口径）"
          f" → 差 {abs(got - 84.85) / 84.85 * 100:.2f}%")
    ok &= abs(got - 84.85) / 84.85 < 0.02

    # ③ attention core 必须与 8B LoRA 版同公式（12*L*hidden*ΣS²）
    a = step_flops(QWEN3_14B, [4096])["attention_core"]
    assert a == 12 * 40 * 5120 * 4096 ** 2, "attention core 公式漂移"
    print("③ attention core 公式 = 12·L·hidden·ΣS²  一致")

    print(f"结论: {'全部通过' if ok else '有项目未通过'}")
    return ok


if __name__ == "__main__":
    import sys
    ok = selfcheck()
    print()
    for s in (2048, 4096, 8192):
        t = step_flops(QWEN3_14B, [s])["total"] / s
        print(f"  cutoff {s:>5}: {t / 1e9:6.2f} GFLOP/token（含 attention core）")
    sys.exit(0 if ok else 1)
