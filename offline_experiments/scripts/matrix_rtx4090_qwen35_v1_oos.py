#!/usr/bin/env python3
"""4090 Qwen3.5 v1 预测器的独立 out-of-sample 验证矩阵。

为什么要做
----------
分组留出（`validate_rtx4090_qwen35_v1_group_holdout.py`）只能在既有 532 组
内部重切，看不到「整个配置组合从未被跑过」的地方。VL 侧的教训很直接：
v1 的 LOO 说吞吐 MAPE 18.6%，真 out-of-sample 是 56.3%，缺陷只有真跑才暴露。

20 组作业，全部落在拟合池的结构空白上：

  A. 多卡 + GC（5 组）——拟合池 48 条 gc 行**全是单卡**，零覆盖。
     补上之后才可能加 `gc × log_gpu` 交叉项（VL v1.1 里这项有效）。
  B. packing + GC（3 组）——零覆盖。
  C. fa2 无 packing + mbs>1（4 组）——**最尖锐的测试**。拟合池 220 条
     fa2-nopack 行全是 mbs=1，而预测器的 `eff_mbs` 对这条路径取 mbs 本身。
     如果 patcher 在不开 packing 时也把 batch 逼成 1，那显存不该随 mbs 涨，
     预测器会大幅高估——这是个可以证伪的预言。
  D. packing + mbs>1（2 组）——检验 `eff_mbs=1` 这个假设。
  E. packing + ga>1（2 组）——ga 在 packing 下零覆盖。
  F. 9B + packing 靠 4 卡 z3（2 组）——拟合池里 9B packing 16 组全 OOM，
     没试过 4 卡 z3 这个最省显存的组合。
  G. cutoff=6144 + packing（2 组）——检验 `1.276 × cutoff` 这个比例在
     未测过的 cutoff 上是否还成立。

复用 `matrix_packing.py` 的执行框架（含全局互斥锁与基线扣除采样），
只替换作业计划，并补上它硬编码为 1 的 `ga`。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path("/wanqing-develop/luowenjing/Param_Recommend/offline_experiments"
            "/artifacts/rtx4090_qwen35_observations_20260825")
sys.path.insert(0, str(_SRC))
import matrix_packing as base  # type: ignore

import yaml  # noqa: E402


OOS_JOBS = [
    # ---- A. 多卡 + GC（拟合池零覆盖，最大结构缺口）----
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="off",
         cutoff=2048, mbs=1, ga=1, gpu_count=2, gc=True, zero="z2",
         note="A_multi_gc_0p8b_g2"),
    dict(model="qwen35_0p8b", dataset_key="multiturn_4096", attn="fa2", pack_mode="off",
         cutoff=4096, mbs=1, ga=1, gpu_count=4, gc=True, zero="z2",
         note="A_multi_gc_0p8b_g4_long"),
    dict(model="qwen35_4b", dataset_key="short_512", attn="fa2", pack_mode="off",
         cutoff=2048, mbs=1, ga=1, gpu_count=2, gc=True, zero="z2",
         note="A_multi_gc_4b_g2"),
    dict(model="qwen35_4b", dataset_key="multiturn_4096", attn="fa2", pack_mode="off",
         cutoff=4096, mbs=1, ga=1, gpu_count=4, gc=True, zero="z2",
         note="A_multi_gc_4b_g4_long"),
    dict(model="qwen35_9b", dataset_key="multiturn_4096", attn="fa2", pack_mode="off",
         cutoff=4096, mbs=1, ga=1, gpu_count=4, gc=True, zero="z2",
         note="A_multi_gc_9b_g4_long"),

    # ---- B. packing + GC（零覆盖）----
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=4096, mbs=1, ga=1, gpu_count=2, gc=True, zero="z2",
         note="B_pack_gc_0p8b_g2"),
    dict(model="qwen35_4b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=4096, mbs=1, ga=1, gpu_count=4, gc=True, zero="z2",
         note="B_pack_gc_4b_g4"),
    dict(model="qwen35_9b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=2048, mbs=1, ga=1, gpu_count=4, gc=True, zero="z3",
         note="B_pack_gc_9b_g4_z3"),

    # ---- C. fa2 无 packing + mbs>1（检验 eff_mbs 假设，最尖锐）----
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="off",
         cutoff=2048, mbs=2, ga=1, gpu_count=1, gc=False, zero="none",
         note="C_fa2_nopack_mbs2_0p8b_g1"),
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="off",
         cutoff=2048, mbs=4, ga=1, gpu_count=2, gc=False, zero="z2",
         note="C_fa2_nopack_mbs4_0p8b_g2"),
    dict(model="qwen35_4b", dataset_key="short_512", attn="fa2", pack_mode="off",
         cutoff=2048, mbs=2, ga=1, gpu_count=2, gc=False, zero="z2",
         note="C_fa2_nopack_mbs2_4b_g2"),
    dict(model="qwen35_0p8b", dataset_key="multiturn_4096", attn="fa2", pack_mode="off",
         cutoff=4096, mbs=2, ga=1, gpu_count=4, gc=False, zero="z2",
         note="C_fa2_nopack_mbs2_0p8b_g4_long"),

    # ---- D. packing + mbs>1（检验 eff_mbs=1 假设）----
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=2048, mbs=2, ga=1, gpu_count=2, gc=False, zero="z2",
         note="D_pack_mbs2_0p8b_g2"),
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=2048, mbs=4, ga=1, gpu_count=4, gc=False, zero="z2",
         note="D_pack_mbs4_0p8b_g4"),

    # ---- E. packing + ga>1（零覆盖）----
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=4096, mbs=1, ga=4, gpu_count=2, gc=False, zero="z2",
         note="E_pack_ga4_0p8b_g2"),
    dict(model="qwen35_4b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=2048, mbs=1, ga=2, gpu_count=4, gc=False, zero="z2",
         note="E_pack_ga2_4b_g4"),

    # ---- F. 9B + packing 靠 4 卡 z3（拟合池里 9B packing 全 OOM）----
    dict(model="qwen35_9b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=2048, mbs=1, ga=1, gpu_count=4, gc=False, zero="z3",
         note="F_9b_pack_g4_z3"),
    dict(model="qwen35_9b", dataset_key="short_512", attn="fa2", pack_mode="pack",
         cutoff=2048, mbs=1, ga=1, gpu_count=4, gc=False, zero="z3",
         note="F_9b_pack_plain_g4_z3"),

    # ---- G. cutoff=6144 + packing（检验 1.276×cutoff 比例外推）----
    dict(model="qwen35_0p8b", dataset_key="short_512", attn="fa2", pack_mode="neat",
         cutoff=6144, mbs=1, ga=1, gpu_count=2, gc=False, zero="z2",
         note="G_cutoff6144_0p8b_g2"),
    dict(model="qwen35_4b", dataset_key="multiturn_4096", attn="fa2", pack_mode="neat",
         cutoff=6144, mbs=1, ga=1, gpu_count=4, gc=False, zero="z2",
         note="G_cutoff6144_4b_g4"),
]


def build_cfg_with_ga(*, ga, **kw):
    """base.build_cfg 把 gradient_accumulation_steps 硬编码为 1，这里补上。"""
    cfg = base.build_cfg(**kw)
    cfg["gradient_accumulation_steps"] = ga
    return cfg


def tag_of(j: dict) -> str:
    return (f"{j['model']}_{j['dataset_key']}_{j['attn']}_{j['pack_mode']}"
            f"_c{j['cutoff']}_mbs{j['mbs']}_ga{j['ga']}"
            f"_g{j['gpu_count']}_gc{int(j['gc'])}_{j['zero']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/qwen35_oos_v1"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="只跑 note 以该字母开头的组，例如 --only C")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.is_file():
        done = {json.loads(l)["tag"] for l in results.open()}

    jobs = OOS_JOBS
    if args.only:
        jobs = [j for j in jobs if j["note"].startswith(args.only)]
    print(f"计划 {len(jobs)} 个 out-of-sample 作业，已完成 {len(done)} 个")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['note']:32} {tag_of(j)}")
        return

    lock = base.acquire_lock()
    try:
        for i, j in enumerate(jobs, 1):
            tag = tag_of(j)
            if tag in done:
                continue
            out_dir = args.out / f"out_{tag}"
            kw = {k: v for k, v in j.items() if k not in ("note", "ga")}
            cfg = build_cfg_with_ga(ga=j["ga"], out_dir=out_dir, **kw)
            yaml_path = args.out / f"cfg_{tag}.yaml"
            yaml_path.write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            t0 = time.time()
            res = base.run_one(yaml_path, j["gpu_count"], args.out)
            elapsed = time.time() - t0
            row = dict(
                tag=tag, note=j["note"],
                model=j["model"], dataset=j["dataset_key"],
                attn=j["attn"], pack_mode=j["pack_mode"], cutoff=j["cutoff"],
                mbs=j["mbs"], ga=j["ga"], gpu_count=j["gpu_count"],
                gc=j["gc"], zero=j["zero"],
                warmup=base.WARMUP, measure=base.MEASURE,
                elapsed_s=round(elapsed, 1),
                **res,
            )
            with results.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            flag = "OOM" if res["oom"] else ("FAIL" if res["exit"] != 0 else "ok")
            print(f"[{i}/{len(jobs)}] {j['note']:30} {flag:<5} "
                  f"peak={res['peak_mib']:>6}MiB tok/s={res['tokens_per_s']} "
                  f"({elapsed:.0f}s)", flush=True)
        print("done")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
