#!/usr/bin/env python3
"""4090 Qwen3.5 v1 预测器（加载冻结 JSON 做纯计算）。

用法：

    from predictor_rtx4090_qwen35_v1 import RTX4090Qwen35Predictor
    p = RTX4090Qwen35Predictor.load()
    r = p.predict({
        "model": "qwen35_4b",
        "dataset": "multiturn_4096",
        "attn": "fa2",
        "pack_mode": "neat",     # off / pack / neat
        "cutoff": 4096,
        "mbs": 1,
        "gpu_count": 4,
        "zero": "z2",
        "gc": False,
    })
    # r["feasible"]   配置本身是否可行（sdpa+packing、单卡 z2/z3 都不可行）
    # r["admitted"]   预测能否装进 24 GiB
    # r["tokens_per_s"]  吞吐预估

输入字段范围（超出即为外推，见 r["warnings"]）：
    model      ∈ {qwen35_0p8b, qwen35_4b, qwen35_9b}
    dataset    ∈ {short_512, multiturn_4096, blind_f1, blind_f2}
                 或显式给 tokens_per_sample=int
    attn       ∈ {fa2, sdpa}
    pack_mode  ∈ {off, pack, neat}
    cutoff     ∈ {512, 2048, 4096, 8192}
    mbs        ∈ {1, 2, 4}
    gpu_count  ∈ {1, 2, 4}
    zero       ∈ {none, z2, z3}
    gc         ∈ {False, True}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPO / "artifacts" / "rtx4090_qwen35_v11_predictor.json"

# 拟合数据实际覆盖的取值。超出只警告，不拒绝——外推方向仍然可用于排序。
TRAINED_RANGE = dict(
    mbs={1, 2, 4},
    gpu_count={1, 2, 4},
    cutoff={512, 2048, 4096, 8192},
    zero={"none", "z2", "z3"},
    attn={"fa2", "sdpa"},
    pack_mode={"off", "pack", "neat"},
)

# 显存预测超过这个值就没有物理意义了，只有"拒绝"这个方向可信。
# 拟合数据的峰值上限是 24 GiB，线性外推到 40 GiB 以上纯属数值发散。
ABSURD_MIB = 40000


class RTX4090Qwen35Predictor:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.card_mib = artifact["card_memory_mib"]
        self.model_meta = artifact["model_meta"]
        self.mem_names = artifact["memory_model"]["feature_names"]
        self.mem_w = np.array([artifact["memory_model"]["coefficients"][n]
                               for n in self.mem_names])
        self.safety = float(artifact["memory_model"]["safety_margin_mib"])
        self.tp_names = artifact["throughput_model"]["feature_names"]
        self.tp_w = np.array([artifact["throughput_model"]["coefficients"][n]
                              for n in self.tp_names])
        from fit_rtx4090_qwen35_v11 import (
            memory_features, throughput_features, is_infeasible,
        )
        self._mem_feats = memory_features
        self._tp_feats = throughput_features
        self._infeasible = is_infeasible

    @classmethod
    def load(cls, path: Path | None = None) -> "RTX4090Qwen35Predictor":
        path = path or DEFAULT_ARTIFACT
        return cls(json.loads(Path(path).read_text()))

    def _normalize(self, cfg: dict) -> dict:
        c = dict(cfg)
        c.setdefault("gc", False)
        c.setdefault("zero", "none")
        c.setdefault("gpu_count", 1)
        c.setdefault("mbs", 1)
        c.setdefault("ga", 1)
        c.setdefault("attn", "fa2")
        c.setdefault("pack_mode", "off")
        c.setdefault("cutoff", 4096)
        return c

    def _warnings(self, c: dict) -> list[str]:
        w = []
        for field, allowed in TRAINED_RANGE.items():
            v = c.get(field)
            if v is not None and v not in allowed:
                w.append(f"{field}={v} 超出拟合覆盖 {sorted(allowed)}，属外推")
        if c["model"] == "qwen35_0p8b":
            # leave-one-model-out 实测：留出 0.8B 后 OOM 召回 0/9
            w.append("0.8B 是拟合池里最小的模型；跨模型外推在留出验证中失败过，"
                     "该模型的预测依赖它自身在池内的数据")
        return w

    def predict(self, cfg: dict) -> dict:
        c = self._normalize(cfg)
        reason = self._infeasible(c)
        if reason:
            return dict(
                feasible=False, admitted=False,
                peak_center_mib=None, peak_upper_mib=None, tokens_per_s=None,
                note=reason, warnings=[],
            )
        warns = self._warnings(c)
        f_mem, _ = self._mem_feats(c)
        center = float(f_mem @ self.mem_w)
        upper = center + self.safety
        admitted = upper + 800 < self.card_mib
        f_tp, _ = self._tp_feats(c)
        tps = float(np.exp(f_tp @ self.tp_w))

        note = None
        if center > ABSURD_MIB:
            note = (f"预测中心 {center:.0f} MiB 已进入数值发散区，"
                    f"数字本身无意义，只有 admitted=False 这个方向可信")
        elif not admitted:
            note = "预测上界离卡容量不足 800 MiB 余量"
        return dict(
            feasible=True, admitted=admitted,
            peak_center_mib=center, peak_upper_mib=upper,
            tokens_per_s=tps, note=note, warnings=warns,
        )


def _demo():
    """自检：跑几个典型点，实测值**按 tag 从观测文件里查**，不手写。

    手写实测值踩过坑：VL 预测器的 demo 里抄错一个 peak，看起来像 4.7 倍误差，
    实际是 0.7%。这里直接查，数字不会漂。
    """
    import fit_rtx4090_qwen35_v1 as F
    p = RTX4090Qwen35Predictor.load()
    obs = {r["tag"]: r for r in F.load_observations()}

    cases = [
        ("最优点：0.8B/neat c8192/4卡z2",
         "qwen35_0p8b_short_512_fa2_neat_c8192_mbs1_g4_z2",
         dict(model="qwen35_0p8b", dataset="short_512", attn="fa2",
              pack_mode="neat", cutoff=8192, mbs=1, gpu_count=4, zero="z2")),
        ("塌陷对照：0.8B/fa2 无packing/4卡z2",
         "qwen35_0p8b_short_512_fa2_off_c2048_mbs1_g4_z2",
         dict(model="qwen35_0p8b", dataset="short_512", attn="fa2",
              pack_mode="off", cutoff=2048, mbs=1, gpu_count=4, zero="z2")),
        ("sdpa 路径：0.8B/sdpa mbs4/4卡z2",
         "qwen35_0p8b_short_512_sdpa_off_c2048_mbs4_g4_z2",
         dict(model="qwen35_0p8b", dataset="short_512", attn="sdpa",
              pack_mode="off", cutoff=2048, mbs=4, gpu_count=4, zero="z2")),
        ("9B 靠 z3 救回：9B/4卡z3",
         "qwen35_9b_short_512_fa2_off_c2048_mbs1_g4_z3",
         dict(model="qwen35_9b", dataset="short_512", attn="fa2",
              pack_mode="off", cutoff=2048, mbs=1, gpu_count=4, zero="z3")),
        ("应拒：4B/neat c4096/单卡（实测 OOM）",
         "qwen35_4b_multiturn_4096_fa2_neat_c4096_mbs1_g1_none",
         dict(model="qwen35_4b", dataset="multiturn_4096", attn="fa2",
              pack_mode="neat", cutoff=4096, mbs=1, gpu_count=1, zero="none")),
        ("配置不可行：sdpa + neat",
         None,
         dict(model="qwen35_0p8b", dataset="short_512", attn="sdpa",
              pack_mode="neat", cutoff=4096, mbs=1, gpu_count=1, zero="none")),
        ("配置不可行：单卡 z2",
         None,
         dict(model="qwen35_4b", dataset="short_512", attn="fa2",
              pack_mode="off", cutoff=512, mbs=1, gpu_count=1, zero="z2")),
        ("配置不可行：fa2 + mbs>1（v1.1 新门）",
         None,
         dict(model="qwen35_0p8b", dataset="short_512", attn="fa2",
              pack_mode="off", cutoff=2048, mbs=4, gpu_count=2, zero="z2")),
    ]

    print(f"{'配置':36} {'预测peak':>8} {'实测peak':>8} {'误差':>8} "
          f"{'放行':>4} {'实况':>5} {'预测t/s':>9} {'实测t/s':>9} {'误差':>7}")
    print("-" * 112)
    for label, tag, cfg in cases:
        r = p.predict(cfg)
        if not r["feasible"]:
            print(f"{label:36} {'—':>8} {'—':>8} {'—':>8} {'否':>4} {'不可行':>5} "
                  f"{'—':>9} {'—':>9} {'—':>7}")
            continue
        o = obs.get(tag)
        if o is None:
            print(f"{label:36} {r['peak_center_mib']:8.0f} {'未测':>8} {'—':>8} "
                  f"{('是' if r['admitted'] else '否'):>4} {'未测':>5} "
                  f"{r['tokens_per_s']:9.0f} {'未测':>9} {'—':>7}")
            continue
        real_oom = o["oom"]
        real_peak = o["peak_mib"]
        mem_err = f"{(r['peak_center_mib']-real_peak)/real_peak*100:+7.1f}%"
        real_tps = o.get("tokens_per_s")
        if real_tps and not real_oom:
            tps_s = f"{real_tps:9.0f}"
            tps_err = f"{(r['tokens_per_s']-real_tps)/real_tps*100:+6.1f}%"
        else:
            tps_s, tps_err = f"{'—':>9}", f"{'—':>7}"
        print(f"{label:36} {r['peak_center_mib']:8.0f} {real_peak:8.0f} {mem_err} "
              f"{('是' if r['admitted'] else '否'):>4} {('OOM' if real_oom else 'ok'):>5} "
              f"{r['tokens_per_s']:9.0f} {tps_s} {tps_err}")


if __name__ == "__main__":
    _demo()
