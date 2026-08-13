#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3 LoRA-aware 逻辑模型 FLOPs 计算器。

严格按《MFU 实验统计原则》§3 实现：
  - 冻结 base Linear：4*M*K*N（forward + dX，不含 dW）
  - 可训练 LoRA adapter：6*M*r*(K+N)
  - attention core：12 * L * hidden * sum(S_j^2)
  - activation recomputation 不进分子
  - lm_head 冻结且非 LoRA target → 4*M*K*N（CCE 用同一 dense-reference）

自校验：用 Qwen3-8B / MBS=8 / GAS=4 / seq=4096 复算，必须逐位等于参考实验
        4986491390394368 FLOPs/step 及其六个分量。
"""

from __future__ import annotations

import json
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


# LoRA target -> (K, N) 该 Linear 的输入/输出维度
def linear_dims(ms: ModelShape) -> dict[str, tuple[int, int]]:
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


def step_flops(
    ms: ModelShape,
    seq_lengths: list[int],
    lora_rank: int,
    lora_targets: tuple[str, ...],
) -> dict[str, int]:
    """一个 optimizer step 的逻辑 FLOPs 分量。

    seq_lengths: 该 step 全部 microbatch 里每条真实序列的长度（padded 布局下即每行长度）。
    """
    dims = linear_dims(ms)
    tokens = sum(seq_lengths)  # 参与 Linear 的 token/row 总数 M
    L = ms.num_layers

    # 冻结 base Linear：forward + dX = 4*M*K*N
    frozen_attn = 4 * tokens * L * sum(dims[n][0] * dims[n][1] for n in ATTN_PROJ)
    frozen_mlp = 4 * tokens * L * sum(dims[n][0] * dims[n][1] for n in MLP_PROJ)
    # lm_head 冻结、非 LoRA target；CCE 用同一 dense-reference
    frozen_lm_head = 4 * tokens * ms.hidden * ms.vocab

    # LoRA adapter：6*M*r*(K+N)
    lora_attn = (
        6 * tokens * lora_rank * L
        * sum(dims[n][0] + dims[n][1] for n in ATTN_PROJ if n in lora_targets)
    )
    lora_mlp = (
        6 * tokens * lora_rank * L
        * sum(dims[n][0] + dims[n][1] for n in MLP_PROJ if n in lora_targets)
    )

    # attention core：12 * L * hidden * sum(S^2)
    attn_core = 12 * L * ms.hidden * sum(s * s for s in seq_lengths)

    parts = {
        "frozen_attn_projection": frozen_attn,
        "frozen_mlp": frozen_mlp,
        "frozen_lm_head": frozen_lm_head,
        "lora_attn": lora_attn,
        "lora_mlp": lora_mlp,
        "attention_core": attn_core,
    }
    parts["total"] = sum(parts.values())
    return parts


QWEN3_8B = ModelShape(
    num_layers=36, hidden=4096, num_heads=32, num_kv_heads=8,
    head_dim=128, intermediate=12288, vocab=151936,
)

ALL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# 参考实验（hpo_fa3_cce_lorafusion，MBS=8/GAS=4/seq=4096）的权威分量
REFERENCE = {
    "frozen_attn_projection": 791648371998720,
    "frozen_mlp": 2849934139195392,
    "frozen_lm_head": 326280075542528,
    "lora_attn": 24120536334336,
    "lora_mlp": 44530220924928,
    "attention_core": 949978046398464,
    "total": 4986491390394368,
}


def selfcheck() -> bool:
    got = step_flops(QWEN3_8B, [4096] * 32, lora_rank=32, lora_targets=ALL_TARGETS)
    ok = got == REFERENCE
    print("=== FLOPs 公式自校验 (MBS=8 x GAS=4 x seq4096 = 32 seqs) ===")
    for k in REFERENCE:
        mark = "OK " if got[k] == REFERENCE[k] else "MISMATCH"
        print(f"  {mark} {k:26s} got={got[k]:>20d} ref={REFERENCE[k]:>20d}")
    print(f"  结论: {'逐位一致' if ok else '不一致，公式或参数有误'}")
    return ok and segmentation_selfcheck()


def segmentation_selfcheck() -> bool:
    """校验分段口径：把一条长序列切成 n 段后 attention 必须降到 1/n。

    上面那个参考用例是 packing=false 的定长数据，每行恰好一条样本（n=1），
    "整行长度"与"逐样本长度"在该用例下完全相同 —— 所以它对块对角口径错误
    **天然免疫**，曾让一个把整包当单条序列的 bug 通过了自校验（实测 MFU 因此
    虚高 1.63 倍）。这里显式构造 n>1 的用例把那条路堵上。
    """
    # 段数必须整除 pack 长度，否则切分后 token 总数就变了，Linear 分量会跟着变，
    # 测不出"只有 attention 受切分影响"这个性质。
    pack, segments = 16384, 8
    per = pack // segments
    assert per * segments == pack, "段长必须整除 pack 长度"
    whole = step_flops(QWEN3_8B, [pack], 32, ALL_TARGETS)
    split = step_flops(QWEN3_8B, [per] * segments, 32, ALL_TARGETS)

    # Linear 只与 token 总数有关，切分不改变它；只有 attention 是平方项。
    linear_keys = [k for k in whole if k not in ("attention_core", "total")]
    linear_stable = all(whole[k] == split[k] for k in linear_keys)
    attention_ratio = whole["attention_core"] / split["attention_core"]
    ratio_ok = abs(attention_ratio - segments) < 1e-9

    print()
    print(f"=== 分段口径自校验 (1x{pack} vs {segments}x{per}) ===")
    print(f"  {'OK ' if linear_stable else 'MISMATCH'} Linear 分量切分前后不变")
    print(
        f"  {'OK ' if ratio_ok else 'MISMATCH'} attention 比值 = "
        f"{attention_ratio:.6f}，应等于段数 {segments}"
    )
    print(f"  结论: {'分段口径正确' if linear_stable and ratio_ok else '分段口径错误'}")
    return linear_stable and ratio_ok


if __name__ == "__main__":
    import sys
    ok = selfcheck()
    print()
    print("每 token 逻辑 FLOPs:",
          json.dumps(REFERENCE["total"] / (32 * 4096), indent=None))
    sys.exit(0 if ok else 1)
