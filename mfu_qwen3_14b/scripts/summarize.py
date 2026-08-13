#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 14B 4卡各组的 mfu_report.json 汇总成一张对比表 + 并行污染体检。

体检项（多组同机并行跑，GPU 独占但 CPU/IO 共享）：
  - 稳态 step time 的相对标准差 cv：>2% 视为可疑，建议串行重跑
  - 逻辑 FLOPs/step 是否全组一致（不一致 → MFU 不可直接比，直接报警）
  - 显存峰值距安全线的余量
"""
import json
import pathlib
import re
import statistics
import sys

BASE = pathlib.Path("/wanqing-develop/luowenjing/Param_Recommend/mfu_qwen3_14b/results")
CAP = 143771 / 1024          # 单卡容量 GiB
SAFE = 0.95 * CAP            # 安全线

ORDER = [
    ("r0_mbs1_z3_unsloth", "对照:原配置搬4卡 MBS1 ZeRO3 CPU卸载", 4),
    ("r1_mbs1_z3_gpugc",   "R1 同上但激活值不卸载CPU", 4),
    ("r2_mbs2_z2",         "R2 ZeRO2 MBS2 (前推荐组)", 4),
    ("r3_mbs4_z2",         "R3 ZeRO2 MBS4", 4),
    ("r4_mbs8_z2",         "R4 ZeRO2 MBS8", 4),
    ("d_nogc_mbs1_z2",     "D 关梯度检查点 ZeRO2 MBS1", 4),
    ("c_nogc_mbs2_z3_8gpu", "C 关梯度检查点 ZeRO3 MBS2 【8卡】", 8),
    ("a_neatpack_z2",      "A 开neat_packing ZeRO2 (打包)", 4),
]


def latest(cfg):
    for d in reversed(sorted(BASE.glob(f"{cfg}/run_*"))):
        if (d / "mfu_report.json").is_file():
            return d
    return None


def failure_reason(cfg):
    """没有产物时，去日志里找原因，不要只说'没有产物'。"""
    for d in reversed(sorted(BASE.glob(f"{cfg}/run_*"))):
        log = d / "train.log"
        if not log.is_file():
            continue
        txt = log.read_text(errors="replace")
        for pat, why in [
            (r"torch\.OutOfMemoryError|CUDA out of memory", "OOM 爆显存"),
            (r"Killed|Signal 9|SIGKILL", "被 OOM killer 杀掉(CPU 内存)"),
            (r"NCCL.*(timeout|error)", "NCCL 通信失败"),
            (r"bsz should be 1", "neat_packing 要求单卡批量=1"),
            (r"AssertionError", "断言失败"),
            (r"Traceback", "抛异常(见 train.log)"),
        ]:
            if re.search(pat, txt):
                return why
        # 训练结束会打 train_runtime；没打说明还在跑，不能说成失败
        if not re.search(r"train_runtime", txt):
            return "★ 仍在运行中"
        return "跑完但没产出报告(计时文件缺失?)"
    return "没跑起来"


def main():
    rows = []
    for cfg, desc, ngpu in ORDER:
        d = latest(cfg)
        if d is None:
            rows.append(dict(cfg=cfg, desc=desc, ngpu=ngpu, ok=False, why=failure_reason(cfg)))
            continue
        rep = json.loads((d / "mfu_report.json").read_text())
        s = rep["per_step_mfu"]["summary"]
        recs = [r for r in rep["per_step_mfu"]["records"] if r["step"] in set(s["steady_steps"])]
        ts = [r["step_time_s_max"] for r in recs]
        # ★ 体检要用逐步 MFU 的 cv，不能用 step time 的 cv。
        #   packing=false 时 collator 按"该 batch 里最长的那条"补齐，不是补到 cutoff，
        #   所以每步真实 FLOPs 本来就不一样，step time 跟着波动是正常的。
        #   逐步 MFU 已经把分子的波动除掉了，剩下的波动才是真干扰。
        ms = [r["mfu"] for r in recs]
        rows.append(dict(
            cfg=cfg, desc=desc, ngpu=ngpu, ok=True, dir=d.name,
            mfu=s["time_weighted_mfu_percent"], step=s["mean_step_time_s"],
            gap=(s.get("data_balance_diagnostics") or {}).get("straggler_gap_percent_mean", 0.0),
            tps=s["padded_tokens_per_second"], tps_real=s["nonpad_tokens_per_second"],
            pad=s["padding_fraction"] * 100, mem=s.get("peak_mem_allocated_gib_max"),
            flops=s["flops_per_step"], n=len(ts),
            src=(recs[0]["length_source"] if recs and "length_source" in recs[0] else "?"),
            cv=(statistics.stdev(ms) / statistics.mean(ms) * 100) if len(ms) > 1 else 0.0,
            cv_step=(statistics.stdev(ts) / statistics.mean(ts) * 100) if len(ts) > 1 else 0.0,
        ))

    ok = [r for r in rows if r["ok"]]
    W = 124
    print("\n" + "=" * W)
    print(f"{'配置':<40}{'卡':>4}{'MFU%':>8}{'step s':>9}{'非padtok/s':>12}"
          f"{'显存GiB':>9}{'补零%':>8}{'gap%':>7}{'cv%':>7}{'步数':>6}")
    print("-" * W)
    for r in rows:
        if not r["ok"]:
            print(f"{r['desc']:<40}{('—— ' + r['why'] + ' ——'):>56}")
            continue
        print(f"{r['desc']:<40}{r['ngpu']:>4}{r['mfu']:>8.3f}{r['step']:>9.3f}"
              f"{r['tps_real']:>12.1f}{(r['mem'] or 0):>9.1f}{r['pad']:>8.1f}"
              f"{r['gap']:>7.2f}{r['cv']:>7.2f}{r['n']:>6}")
    print("=" * W)
    print("★ MFU 分母含卡数（ΣF/(ΣT·N_GPU·P_peak)），是单卡平均利用率，跨卡数可直接比。")
    print("  『非padtok/s』和『gap%』属 data_balance_diagnostics，是诊断量，不参与 MFU 口径。")

    if not ok:
        print("\n没有任何可用产物。")
        return 1

    flops = {r["flops"] for r in ok}
    if len(flops) == 1:
        print(f"\n[分子一致] 全部 {len(ok)} 组 逻辑 FLOPs/step = {flops.pop()/1e15:.6f} PFLOPs "
              f"→ MFU 可直接比较")
    else:
        print(f"\n[分子不一致 —— 本实验属正常] 出现 {len(flops)} 种 FLOPs/step。")
        print("    原因：packing=false 时 collator 补齐到'该 batch 内最长那条'，")
        print("    MBS 越大补零越多，所以每步 FLOPs 随组而变。")
        print("    ★ MFU 仍然可比：补零的算力是真花掉了，进分子是正确口径。")
        print("      但 MFU 高不等于训得多 —— 补零多的组每 padded token 的信息量更低，")
        print("      所以跨组选型要 MFU 与『非padding token/s』一起看，后者是诊断量。")

    noisy = [r for r in ok if r["cv"] > 2.0]
    print("\n[波动体检] 稳态『逐步 MFU』的相对标准差 cv（分子波动已除掉，剩的是真波动）")
    if noisy:
        for r in noisy:
            print(f"    {r['cfg']:24s} MFU cv={r['cv']:5.2f}%  (step time cv={r['cv_step']:5.2f}%)")
    else:
        print(f"    全部 {len(ok)} 组 cv 均 <2%，稳态干净")

    base = next((r for r in ok if r["cfg"] == ORDER[0][0]), None)
    print("\n[排名] 按 MFU 降序" + (f"，括号内为相对对照组 {base['mfu']:.3f}% 的变化" if base else ""))
    for i, r in enumerate(sorted(ok, key=lambda x: -x["mfu"]), 1):
        tail = ""
        if base:
            tail = (f"  (MFU {r['mfu'] - base['mfu']:+.2f} 点, 有效吞吐 "
                    f"{(r['tps_real'] / base['tps_real'] - 1) * 100:+.1f}%)")
        print(f"    {i}. {r['desc']:<40}{r['mfu']:>8.3f}%{tail}")

    print(f"\n[显存] 单卡容量 {CAP:.1f} GiB，安全线 0.95x = {SAFE:.1f} GiB（取各组 4 卡里最高的那张）")
    for r in sorted(ok, key=lambda x: -(x["mem"] or 0)):
        if r["mem"]:
            head = SAFE - r["mem"]
            flag = "  ← 余量不足 5GiB，换数据分布就可能爆" if head < 5 else ""
            print(f"    {r['desc']:<40}{r['mem']:>7.1f} GiB  余量 {head:>6.1f} GiB{flag}")

    print(f"\n[padding 浪费] cutoff=4096 但数据中位长度仅 615，算力有很大比例花在 padding 上")
    for r in ok:
        print(f"    {r['desc']:<40}padding 占 {r['pad']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
