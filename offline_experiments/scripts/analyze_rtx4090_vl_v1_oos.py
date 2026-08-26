#!/usr/bin/env python3
"""比较 v1 预测器在 out-of-sample 12 组作业上的实测 vs 预测。

输入：/tmp/vl4090_oos_v1/results.jsonl（由 matrix_rtx4090_vl_v1_oos.py 生成）
输出：
  1. 每行的对照表：actual peak / center pred / upper / admit / 实测 vs 预测 tok/s
  2. 全局：
     - OOM 召回（预测 admit=False 且 actual OOM 的比例）
     - 显存 RMSE / MAE / 保守偏差方向
     - 吞吐 MAPE（仅成功行）
     - 分缺口的细分
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from predictor_rtx4090_vl_v1 import RTX4090VLPredictor

RESULTS = Path("/tmp/vl4090_oos_v1/results.jsonl")


def main():
    if not RESULTS.is_file():
        print(f"结果文件缺失：{RESULTS}"); return
    rows = [json.loads(l) for l in RESULTS.open()]
    p = RTX4090VLPredictor.load()

    print(f"{'#':>2} {'note':22} {'actual':>10}  {'center':>7}  {'upper':>7}  "
          f"{'admit':>6}/{'reality':>7}  {'tok/s pred':>10}  {'tok/s act':>9}")
    print("-" * 105)

    mem_errs, tps_pairs, oom_hits = [], [], []
    fr_ok, wrong_admit_oom = 0, 0
    for i, r in enumerate(rows, 1):
        cfg = dict(model=r["model"], dataset=r["dataset"], mbs=r["mbs"],
                   gpu_count=r["gpu_count"], gc=r["gc"], zero=r["zero"])
        pred = p.predict(cfg)
        actual_oom = r["oom"]
        actual_peak = r["peak_mib"] if not actual_oom else None
        actual_tps = r.get("tokens_per_s")
        note = r.get("note", "")
        # 显存
        if actual_peak is not None:
            mem_errs.append((cfg, r, pred["peak_center_mib"], actual_peak))
        # 吞吐
        if not actual_oom and actual_tps:
            tps_pairs.append((pred["tokens_per_s"], actual_tps))
        # 准入
        reality = "OOM" if actual_oom else "ok"
        if actual_oom:
            if not pred["admitted"]:
                oom_hits.append(True)
            else:
                oom_hits.append(False)
                wrong_admit_oom += 1
        else:
            if not pred["admitted"] and (actual_peak or 0) < 22000:
                fr_ok += 1

        peak_str = f"{actual_peak:>10}" if actual_peak is not None else "     OOM  "
        tps_act = f"{actual_tps:9.0f}" if actual_tps else "     n/a "
        print(f"{i:>2} {note:22} {peak_str}  {pred['peak_center_mib']:7.0f}  "
              f"{pred['peak_upper_mib']:7.0f}  {str(pred['admitted']):>6}/{reality:>7}  "
              f"{pred['tokens_per_s']:10.0f}  {tps_act}")

    print()
    print("========== 汇总 ==========")
    # 显存
    if mem_errs:
        deltas = np.array([c - a for (_, _, c, a) in mem_errs])
        actuals = np.array([a for (_, _, _, a) in mem_errs])
        print(f"显存中心（成功行 n={len(mem_errs)}）:")
        print(f"  RMSE = {float(np.sqrt(np.mean(deltas**2))):.0f} MiB")
        print(f"  MAE  = {float(np.mean(np.abs(deltas))):.0f} MiB")
        print(f"  平均偏差 = {float(np.mean(deltas)):+.0f} MiB（正=预测偏高）")
        print(f"  最大低估 = {float(np.min(deltas)):+.0f} MiB  最大高估 = {float(np.max(deltas)):+.0f} MiB")
    # 吞吐
    if tps_pairs:
        preds = np.array([p for (p, _) in tps_pairs])
        actuals = np.array([a for (_, a) in tps_pairs])
        mape = float(np.mean(np.abs(preds - actuals) / actuals))
        print(f"吞吐（成功行 n={len(tps_pairs)}）:")
        print(f"  MAPE = {mape*100:.1f}%")
        print(f"  最好: 预测 {preds[np.argmin(np.abs(preds-actuals)/actuals)]:.0f} 实测 "
              f"{actuals[np.argmin(np.abs(preds-actuals)/actuals)]:.0f}")
        print(f"  最差: 预测 {preds[np.argmax(np.abs(preds-actuals)/actuals)]:.0f} 实测 "
              f"{actuals[np.argmax(np.abs(preds-actuals)/actuals)]:.0f}")
    # 准入
    total_oom = sum(1 for r in rows if r["oom"])
    print(f"准入决策:")
    print(f"  OOM 召回 = {sum(oom_hits)}/{total_oom}"
          f" = {sum(oom_hits)/max(total_oom,1)*100:.0f}%")
    print(f"  错误接受 OOM 配置 = {wrong_admit_oom}")
    print(f"  错拒 ok 配置（peak<22GiB）= {fr_ok}")


if __name__ == "__main__":
    main()
