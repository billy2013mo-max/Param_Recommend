#!/usr/bin/env python3
"""4090 VL v1 预测器（加载冻结的 JSON，做纯计算）。

用法：

    from predictor_rtx4090_vl_v1 import RTX4090VLPredictor
    p = RTX4090VLPredictor.load()
    r = p.predict({
        "model": "qwen25vl3b",
        "dataset": "blind_f2",   # 或直接给 tokens_per_sample=1739
        "mbs": 2,
        "gc": False,
        "zero": "z2",
        "gpu_count": 2,
    })
    # r = {
    #   "peak_center_mib": ...,
    #   "peak_upper_mib": ...,
    #   "admitted": True/False,
    #   "tokens_per_s": ...,
    # }

只对训练配置里的以下字段敏感：
    model ∈ {qwen25vl3b, qwen3vl4b}
    dataset ∈ {blind_f1, blind_f2, blind_f4, calib_f2}  或 tokens_per_sample=int
    mbs ∈ {1, 2, 4}         （模型对更大 mbs 是外推，不保证）
    gc ∈ {False, True}
    zero ∈ {none, z2, z3}
    gpu_count ∈ {1, 2, 4}

cutoff_len 恒定 4096（拟合数据背景），不作为变量。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPO / "artifacts" / "rtx4090_vl_v1_predictor.json"


class RTX4090VLPredictor:
    def __init__(self, artifact: dict):
        self.artifact = artifact
        self.card_mib = artifact["card_memory_mib"]
        self.model_meta = artifact["model_meta"]
        self.dataset_tokens = artifact["dataset_tokens"]
        self.mem_names = artifact["memory_model"]["feature_names"]
        self.mem_w = np.array([artifact["memory_model"]["coefficients"][n]
                               for n in self.mem_names])
        self.safety = float(artifact["memory_model"]["safety_margin_mib"])
        self.tp_names = artifact["throughput_model"]["feature_names"]
        self.tp_w = np.array([artifact["throughput_model"]["coefficients"][n]
                              for n in self.tp_names])
        # 与拟合脚本共用一份特征函数：动态导入避免耦合
        from fit_rtx4090_vl_v1 import memory_features, throughput_features
        self._mem_feats = memory_features
        self._tp_feats = throughput_features

    @classmethod
    def load(cls, path: Path | None = None) -> "RTX4090VLPredictor":
        path = path or DEFAULT_ARTIFACT
        return cls(json.loads(Path(path).read_text()))

    def _normalize(self, cfg: dict) -> dict:
        """填默认值，保证特征函数不缺参数。"""
        c = dict(cfg)
        c.setdefault("gc", False)
        c.setdefault("zero", "none")
        c.setdefault("gpu_count", 1)
        return c

    def predict(self, cfg: dict) -> dict:
        c = self._normalize(cfg)
        # 显存
        f_mem, _ = self._mem_feats(c)
        assert list(_[0] if isinstance(_, tuple) else _
                    for _ in [self.mem_names]) == [self.mem_names]
        center = float(f_mem @ self.mem_w)
        upper = center + self.safety
        admitted = upper + 800 < self.card_mib  # 与拟合脚本对齐的 headroom
        # 吞吐
        f_tp, _ = self._tp_feats(c)
        log_tps = float(f_tp @ self.tp_w)
        tps = float(np.exp(log_tps))
        return dict(
            peak_center_mib=center,
            peak_upper_mib=upper,
            admitted=admitted,
            tokens_per_s=tps,
            note=None if admitted else "predicted upper >= card memory - 800 MiB",
        )


def _demo():
    """自检：跑几个典型点，看预测和实测方向是否合理。"""
    p = RTX4090VLPredictor.load()
    cases = [
        # (label, cfg, 实测 peak, 实测 tok/s)
        ("3B/blind_f1/单卡/mbs1/gc",
         dict(model="qwen25vl3b", dataset="blind_f1", mbs=1, gc=True),
         8861, 1265),
        ("3B/blind_f2/2 卡 z2/mbs2",
         dict(model="qwen25vl3b", dataset="blind_f2", mbs=2, gpu_count=2, zero="z2"),
         21265, 5937),
        ("4B/blind_f2/4 卡 z2/mbs1",
         dict(model="qwen3vl4b", dataset="blind_f2", mbs=1, gpu_count=4, zero="z2"),
         21659, 7896),
        ("4B/calib_f2/2 卡 z2/mbs2（实测 OOM）",
         dict(model="qwen3vl4b", dataset="calib_f2", mbs=2, gpu_count=2, zero="z2"),
         "OOM(24209)", "-"),
    ]
    print(f"{'配置':40}  {'center':>7}  {'upper':>7}  {'admit':>6}  {'tok/s':>7}  |  实测")
    for label, cfg, real_peak, real_tps in cases:
        r = p.predict(cfg)
        print(f"{label:40}  {r['peak_center_mib']:7.0f}  {r['peak_upper_mib']:7.0f}"
              f"  {str(r['admitted']):>6}  {r['tokens_per_s']:7.0f}  |  "
              f"peak={real_peak}  tps={real_tps}")


if __name__ == "__main__":
    _demo()
