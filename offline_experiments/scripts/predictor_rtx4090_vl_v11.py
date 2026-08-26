#!/usr/bin/env python3
"""4090 VL v1.1 预测器：加载冻结 JSON 做预测。

相对 v1 的改动：
  - 吞吐特征加了 `gc × log_gpu` 交叉项
  - 数据从 108 行扩到 140 行（+OOS 10 + 补数 22）

用法与 v1 完全相同，见 predictor_rtx4090_vl_v1.py 的 docstring。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = REPO / "artifacts" / "rtx4090_vl_v11_predictor.json"


class RTX4090VLPredictorV11:
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
        # v1.1 显存特征沿用 v1；吞吐用 v1.1 的
        from fit_rtx4090_vl_v1 import memory_features
        from fit_rtx4090_vl_v11 import throughput_features
        self._mem_feats = memory_features
        self._tp_feats = throughput_features

    @classmethod
    def load(cls, path: Path | None = None) -> "RTX4090VLPredictorV11":
        path = path or DEFAULT_ARTIFACT
        return cls(json.loads(Path(path).read_text()))

    def _normalize(self, cfg: dict) -> dict:
        c = dict(cfg)
        c.setdefault("gc", False); c.setdefault("zero", "none")
        c.setdefault("gpu_count", 1)
        return c

    def predict(self, cfg: dict) -> dict:
        c = self._normalize(cfg)
        # 前置门（v1 未做）：DeepSpeed 不支持单卡 z2/z3，避免 admit 一个跑不起来的配置
        if c["gpu_count"] == 1 and c["zero"] in ("z2", "z3"):
            return dict(
                peak_center_mib=float("nan"),
                peak_upper_mib=float("nan"),
                admitted=False,
                tokens_per_s=float("nan"),
                note="DeepSpeed 不支持单卡 zero-2/3；配置本身不可行",
            )
        f_mem, _ = self._mem_feats(c)
        center = float(f_mem @ self.mem_w); upper = center + self.safety
        admitted = upper + 800 < self.card_mib
        f_tp, _ = self._tp_feats(c)
        tps = float(np.exp(f_tp @ self.tp_w))
        return dict(
            peak_center_mib=center,
            peak_upper_mib=upper,
            admitted=admitted,
            tokens_per_s=tps,
            note=None if admitted else "predicted upper too close to card limit",
        )


def _demo():
    p = RTX4090VLPredictorV11.load()
    cases = [
        ("3B/blind_f1/单卡/mbs1/gc",
         dict(model="qwen25vl3b", dataset="blind_f1", mbs=1, gc=True), 8861, 1265),
        ("3B/blind_f2/2 卡 z2/mbs2",
         dict(model="qwen25vl3b", dataset="blind_f2", mbs=2, gpu_count=2, zero="z2"),
         21265, 5937),
        ("4B/blind_f2/4 卡 z2/mbs1",
         dict(model="qwen3vl4b", dataset="blind_f2", mbs=1, gpu_count=4, zero="z2"),
         21659, 7896),
        ("4B/calib_f2/2 卡 z2/mbs2（应拒）",
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
