#!/usr/bin/env python3
"""比较 4090 Qwen3.5 v1 预测器在 20 组 out-of-sample 作业上的实测 vs 冻结预测。

预测在跑作业**之前**就冻结到
`artifacts/rtx4090_qwen35_v1_oos_frozen_predictions.json`，
本脚本只读那份冻结值，不重新调用预测器——避免"看到结果再改模型再报成绩"。

输出：
  1. 逐组对照表
  2. 按缺口类别（A~G）汇总
  3. 全局：准入正确率、显存 RMSE/相对误差、吞吐 MAPE
  4. 逐条检验预注册的两个假设（eff_mbs 在 fa2 下取 mbs / packing 下取 1）
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "artifacts" / "rtx4090_qwen35_v1_oos_frozen_predictions.json"
RESULTS = Path("/tmp/qwen35_oos_v1/results.jsonl")

GAP_NAMES = {
    "A": "多卡 + GC",
    "B": "packing + GC",
    "C": "fa2 无packing + mbs>1",
    "D": "packing + mbs>1",
    "E": "packing + ga>1",
    "F": "9B packing 靠 4卡z3",
    "G": "cutoff=6144 外推",
}


def main():
    if not RESULTS.is_file():
        print(f"结果文件缺失：{RESULTS}")
        return
    frozen = {r["tag"]: r for r in json.loads(FROZEN.read_text())}
    rows = [json.loads(l) for l in RESULTS.open()]

    print(f"{'组':30} {'预测peak':>8} {'实测peak':>8} {'误差':>8} "
          f"{'放行':>4} {'实况':>5} {'预测t/s':>9} {'实测t/s':>9} {'误差':>8}")
    print("-" * 104)

    per_gap = defaultdict(lambda: dict(mem_rel=[], tps_rel=[], admit_ok=0, n=0,
                                       wrong_admit=0, false_reject=0, fails=[]))
    g_mem, g_tps = [], []
    admit_correct = admit_total = wrong_admit = false_reject = 0
    framework_fails = []

    for r in rows:
        f = frozen.get(r["tag"])
        if f is None:
            print(f"{r.get('note','?'):30} 找不到冻结预测，跳过")
            continue
        gap = f["note"][0]
        st = per_gap[gap]
        st["n"] += 1
        real_oom = r["oom"]
        real_fail = r["exit"] != 0 and not real_oom
        real_peak = r["peak_mib"]
        real_tps = r.get("tokens_per_s")
        pc, pa, pt = f["pred_center"], f["pred_admitted"], f["pred_tps"]

        if real_fail:
            framework_fails.append((f["note"], r))
            st["fails"].append(f["note"])
            print(f"{f['note']:30} {pc:8.0f} {real_peak:8.0f} {'—':>8} "
                  f"{('是' if pa else '否'):>4} {'FAIL':>5} {pt:9.0f} {'—':>9} {'—':>8}")
            continue

        # 准入判定
        admit_total += 1
        if real_oom:
            if not pa:
                admit_correct += 1
                st["admit_ok"] += 1
            else:
                wrong_admit += 1
                st["wrong_admit"] += 1
        else:
            if pa:
                admit_correct += 1
                st["admit_ok"] += 1
            else:
                false_reject += 1
                st["false_reject"] += 1

        # 显存误差只在成功行上算（OOM 行的 peak 是崩溃时的卡容量，不是真实需求）
        if not real_oom and real_peak:
            rel = (pc - real_peak) / real_peak
            g_mem.append(rel); st["mem_rel"].append(rel)
            mem_s = f"{rel*100:+7.1f}%"
        else:
            mem_s = f"{'—':>8}"
        if not real_oom and real_tps:
            trel = (pt - real_tps) / real_tps
            g_tps.append(trel); st["tps_rel"].append(trel)
            tps_s, terr = f"{real_tps:9.0f}", f"{trel*100:+7.1f}%"
        else:
            tps_s, terr = f"{'—':>9}", f"{'—':>8}"

        print(f"{f['note']:30} {pc:8.0f} {real_peak:8.0f} {mem_s} "
              f"{('是' if pa else '否'):>4} {('OOM' if real_oom else 'ok'):>5} "
              f"{pt:9.0f} {tps_s} {terr}")

    print()
    print("========== 按缺口类别汇总 ==========")
    print(f"{'类别':26} {'n':>3} {'准入对':>5} {'错放':>4} {'误拒':>4} "
          f"{'框架失败':>7} {'显存相对MAE':>11} {'吞吐MAPE':>9}")
    print("-" * 82)
    for gap in sorted(per_gap):
        st = per_gap[gap]
        mm = (f"{np.mean(np.abs(st['mem_rel']))*100:10.1f}%"
              if st["mem_rel"] else f"{'n/a':>11}")
        tt = (f"{np.mean(np.abs(st['tps_rel']))*100:8.1f}%"
              if st["tps_rel"] else f"{'n/a':>9}")
        print(f"{gap+'. '+GAP_NAMES[gap]:26} {st['n']:>3} {st['admit_ok']:>5} "
              f"{st['wrong_admit']:>4} {st['false_reject']:>4} "
              f"{len(st['fails']):>7} {mm} {tt}")

    print()
    print("========== 全局 ==========")
    print(f"  作业总数 = {len(rows)}，其中框架失败 {len(framework_fails)} 个"
          f"（不计入准入统计）")
    print(f"  准入正确率 = {admit_correct}/{admit_total} "
          f"= {admit_correct/max(admit_total,1)*100:.1f}%")
    print(f"    错误放行 OOM 配置 = {wrong_admit}  ← 这是唯一不可接受的错误方向")
    print(f"    误拒可跑配置     = {false_reject}")
    if g_mem:
        g = np.array(g_mem)
        print(f"  显存（成功行 n={len(g)}）：相对 MAE = {np.mean(np.abs(g))*100:.1f}%  "
              f"平均偏差 = {np.mean(g)*100:+.1f}%  "
              f"最大高估 = {g.max()*100:+.1f}%  最大低估 = {g.min()*100:+.1f}%")
    if g_tps:
        t = np.array(g_tps)
        print(f"  吞吐（成功行 n={len(t)}）：MAPE = {np.mean(np.abs(t))*100:.1f}%  "
              f"平均偏差 = {np.mean(t)*100:+.1f}%")

    if framework_fails:
        print()
        print("========== 框架失败明细（配置层面跑不起来，应考虑加前置门）==========")
        for note, r in framework_fails:
            tail = (r.get("log_tail") or "")[-260:].replace("\n", " ")
            print(f"  {note}: exit={r['exit']} peak={r['peak_mib']}")
            if tail:
                print(f"      日志尾: …{tail}")

    print()
    print("========== 预注册假设检验 ==========")
    print("假设 1：fa2 无 packing 时有效微批 = mbs（显存随 mbs 线性涨）")
    print("  → 看 C 组。若实测 peak 不随 mbs 涨而预测涨了，假设伪，需改 eff_mbs。")
    c = per_gap.get("C")
    if c and c["mem_rel"]:
        m = np.array(c["mem_rel"])
        verdict = ("成立（预测未系统性高估）" if np.mean(m) < 0.15
                   else f"**可能伪**：C 组平均高估 {np.mean(m)*100:+.1f}%")
        print(f"  C 组平均偏差 {np.mean(m)*100:+.1f}% → {verdict}")
    print("假设 2：fa2 + packing 时有效微批 = 1（显存不随 mbs 涨）")
    print("  → 看 D 组。若实测 peak 随 mbs 明显上涨而预测持平，假设伪。")
    d = per_gap.get("D")
    if d and d["mem_rel"]:
        m = np.array(d["mem_rel"])
        verdict = ("成立" if np.mean(np.abs(m)) < 0.20
                   else f"**可能伪**：D 组相对 MAE {np.mean(np.abs(m))*100:.1f}%")
        print(f"  D 组平均偏差 {np.mean(m)*100:+.1f}%，相对 MAE "
              f"{np.mean(np.abs(m))*100:.1f}% → {verdict}")


if __name__ == "__main__":
    main()
