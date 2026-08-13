#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把所有 run 的 mfu_report.json 汇总成一张对比表，并做并行污染体检。

体检项（因为多组是同机并行跑的，GPU 独占但 CPU/IO 共享）：
  - 稳态 step time 的相对标准差 cv：偏大说明受了邻居干扰
  - max-min 极差占均值的比例
  - 逻辑 FLOPs/step 是否全组一致（不一致则 MFU 不可比，直接报警）
"""
import json
import pathlib
import statistics
import sys

BASE = pathlib.Path("/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_8b/results")
ORDER = ["control_maxfull", "best_tput", "ship_mbs16", "ship_mbs32", "nogc_mbs4"]
DESC = {
    "control_maxfull": "对照:文档配置 MBS8 CPU卸载 dr0.1",
    "best_tput":       "上轮最优 MBS16 GPU-GC dr0.0",
    "ship_mbs16":      "可上线 MBS16 GPU-GC dr0.1",
    "ship_mbs32":      "MBS32 GAS1 GPU-GC dr0.1",
    "nogc_mbs4":       "不开GC MBS4 GAS8 dr0.1",
}


def latest(cfg):
    runs = sorted(BASE.glob(f"{cfg}/run_*"))
    for d in reversed(runs):
        if (d / "mfu_report.json").is_file():
            return d
    return None


def main():
    rows = []
    for cfg in ORDER:
        d = latest(cfg)
        if d is None:
            rows.append({"cfg": cfg, "missing": True})
            continue
        rep = json.loads((d / "mfu_report.json").read_text())
        s = rep["per_step_mfu"]["summary"]
        # 方差只能在稳态窗口上算：step1 是 warmup，混进来会把 cv 撑大
        steady = set(s["steady_steps"])
        recs = [r for r in rep["per_step_mfu"]["records"] if r["step"] in steady]
        ts = [r["optimizer_step_time_s"] for r in recs]
        # 显存不在 report 里，得回原始计时文件读（同样只取稳态）
        raw = [json.loads(l) for l in (d / "clean_timing.jsonl").read_text().splitlines() if l.strip()]
        mem = [x["peak_mem_allocated_gib"] for x in raw
               if x["step"] in steady and x.get("peak_mem_allocated_gib")]
        cv = statistics.stdev(ts) / statistics.mean(ts) if len(ts) > 1 else 0.0
        rows.append({
            "cfg": cfg, "dir": d.name, "missing": False,
            "mfu": s["time_weighted_mfu_percent"], "step": s["mean_step_time_s"],
            "tps": s["padded_tokens_per_second"], "flops": s["flops_per_step"],
            "mem": max(mem) if mem else None,
            "cv": cv * 100, "spread": (max(ts) - min(ts)) / statistics.mean(ts) * 100,
            "n": len(ts),
        })

    ok = [r for r in rows if not r["missing"]]
    print("\n" + "=" * 108)
    print(f"{'配置':<34}{'MFU%':>8}{'step s':>9}{'token/s':>11}{'显存GiB':>9}{'cv%':>7}{'极差%':>7}{'步数':>6}")
    print("-" * 108)
    for r in rows:
        if r["missing"]:
            print(f"{DESC.get(r['cfg'], r['cfg']):<34}{'—— 没有产物 ——':>40}")
            continue
        print(f"{DESC[r['cfg']]:<34}{r['mfu']:>8.3f}{r['step']:>9.3f}{r['tps']:>11.1f}"
              f"{(r['mem'] or 0):>9.1f}{r['cv']:>7.2f}{r['spread']:>7.2f}{r['n']:>6}")
    print("=" * 108)

    if not ok:
        print("\n没有任何可用产物。")
        return

    # 分子一致性：这是 MFU 可比的前提
    flops = {r["flops"] for r in ok}
    if len(flops) == 1:
        print(f"\n[分子一致] 全部 {len(ok)} 组 逻辑 FLOPs/step = {flops.pop()} → MFU 可直接比较")
    else:
        print(f"\n[!! 分子不一致 !!] 出现 {len(flops)} 种 FLOPs/step，MFU 不可直接比较：")
        for r in ok:
            print(f"    {r['cfg']}: {r['flops']}")

    # 并行污染体检
    noisy = [r for r in ok if r["cv"] > 2.0]
    print("\n[并行污染体检] 稳态 step time 相对标准差 cv > 2% 视为可疑")
    if noisy:
        for r in noisy:
            print(f"    可疑: {r['cfg']}  cv={r['cv']:.2f}%  极差={r['spread']:.2f}% → 建议串行重跑确认")
    else:
        print(f"    全部 {len(ok)} 组 cv 均 <2%，稳态干净，并行未造成可见干扰")

    # 排名 + 相对基线
    base = next((r for r in ok if r["cfg"] == "control_maxfull"), None)
    print("\n[排名] 按 MFU 降序" + (f"，括号内为相对文档配置 {base['mfu']:.3f}% 的提升" if base else ""))
    for i, r in enumerate(sorted(ok, key=lambda x: -x["mfu"]), 1):
        tail = ""
        if base:
            tail = (f"  ({r['mfu'] - base['mfu']:+.3f} 个点, 吞吐 "
                    f"{(r['tps'] / base['tps'] - 1) * 100:+.1f}%)")
        print(f"    {i}. {DESC[r['cfg']]:<34}{r['mfu']:>8.3f}%{tail}")

    # 显存安全线
    cap = 143771 / 1024
    print(f"\n[显存] 卡容量 {cap:.1f} GiB，安全线 0.95x = {0.95 * cap:.1f} GiB")
    for r in sorted(ok, key=lambda x: -(x["mem"] or 0)):
        if r["mem"]:
            head = 0.95 * cap - r["mem"]
            flag = "  ← 余量不足 5GiB，有爆卡风险" if head < 5 else ""
            print(f"    {DESC[r['cfg']]:<34}{r['mem']:>7.1f} GiB  余量 {head:>6.1f} GiB{flag}")


if __name__ == "__main__":
    sys.exit(main())
