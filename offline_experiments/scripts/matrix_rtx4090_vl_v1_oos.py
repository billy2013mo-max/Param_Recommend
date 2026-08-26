#!/usr/bin/env python3
"""4090 VL 预测器 v1 的独立 out-of-sample 验证矩阵。

12 组作业，全部在拟合池外的配置组合上，用来检验：
  - 显存中心和上界的绝对误差
  - 准入决策的正确性
  - 吞吐 MAPE

配置来自 leave-group-out 揭示的结构缺口：
  A. 多卡 + gc=on   （拟合池 108 行里 0 覆盖）
  B. 单卡 gc=off   （baseline，拟合池 0 覆盖）
  C. mbs=3 内插    （拟合池只覆盖 mbs∈{1,2,4}）
  D. mbs=8 外推    （测极端上界外推）
  E. 单卡 z2 / z3  （拟合池 0 覆盖）

复用 rtx4090_vl_observations_20260823/matrix.py 的执行框架，只替换
job 计划和输出目录。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 复用原矩阵所有基础函数
_SRC = Path("/wanqing-develop/luowenjing/Param_Recommend/offline_experiments"
            "/artifacts/rtx4090_vl_observations_20260823")
sys.path.insert(0, str(_SRC))
import matrix as base  # type: ignore

import yaml  # noqa: E402


OOS_JOBS = [
    # A. 多卡 + gc=on（结构缺口最大）
    dict(model="qwen25vl3b", dataset="blind_f1", mbs=1, gpu_count=2, gc=True,  zero="z2", note="multi_gc_on_easy"),
    dict(model="qwen25vl3b", dataset="blind_f4", mbs=1, gpu_count=2, gc=True,  zero="z2", note="multi_gc_on_tight_3b"),
    dict(model="qwen3vl4b",  dataset="blind_f1", mbs=2, gpu_count=2, gc=True,  zero="z2", note="multi_gc_on_4b_mbs2"),
    dict(model="qwen3vl4b",  dataset="blind_f2", mbs=1, gpu_count=2, gc=True,  zero="z2", note="multi_gc_on_tight_4b"),
    dict(model="qwen3vl4b",  dataset="blind_f4", mbs=1, gpu_count=4, gc=True,  zero="z2", note="multi_gc_on_4b_f4_rescue"),
    # B. 单卡 gc=off（baseline never seen）
    dict(model="qwen25vl3b", dataset="blind_f1", mbs=1, gpu_count=1, gc=False, zero="none", note="single_gc_off_3b"),
    dict(model="qwen3vl4b",  dataset="blind_f1", mbs=1, gpu_count=1, gc=False, zero="none", note="single_gc_off_4b"),
    # C. mbs=3 内插
    dict(model="qwen25vl3b", dataset="blind_f2", mbs=3, gpu_count=2, gc=False, zero="z2", note="mbs3_interior_3b"),
    dict(model="qwen3vl4b",  dataset="blind_f1", mbs=3, gpu_count=4, gc=False, zero="z2", note="mbs3_interior_4b"),
    # D. mbs=8 外推
    dict(model="qwen25vl3b", dataset="blind_f1", mbs=8, gpu_count=2, gc=True,  zero="z2", note="mbs8_extrap_3b"),
    # E. 单卡 z2 / z3
    dict(model="qwen25vl3b", dataset="blind_f1", mbs=2, gpu_count=1, gc=True,  zero="z2", note="single_z2_3b"),
    dict(model="qwen3vl4b",  dataset="blind_f1", mbs=1, gpu_count=1, gc=True,  zero="z3", note="single_z3_4b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/vl4090_oos_v1"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.is_file():
        done = {json.loads(l)["tag"] for l in results.open()}

    jobs = OOS_JOBS
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"计划 {len(jobs)} 个 out-of-sample 作业，已完成 {len(done)} 个")
    if args.dry_run:
        for j in jobs:
            print(" ", j)
        return

    lock = base.acquire_lock()
    try:
        for i, j in enumerate(jobs, 1):
            tag = (f"{j['model']}_{j['dataset']}_mbs{j['mbs']}"
                   f"_g{j['gpu_count']}_gc{int(j['gc'])}_{j['zero']}")
            if tag in done:
                continue
            out_dir = args.out / f"out_{tag}"
            cfg = base.build_cfg(
                model=j["model"], dataset=j["dataset"], mbs=j["mbs"],
                gpu_count=j["gpu_count"], gc=j["gc"], zero=j["zero"],
                out_dir=out_dir,
            )
            yaml_path = args.out / f"cfg_{tag}.yaml"
            yaml_path.write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            t0 = time.time()
            res = base.run_one(yaml_path, j["gpu_count"], args.out)
            elapsed = time.time() - t0
            row = dict(
                tag=tag, note=j.get("note"),
                **{k: v for k, v in j.items() if k != "note"},
                frames=base.DATASETS[j["dataset"]]["frames"],
                cutoff_len=base.CUTOFF,
                warmup=base.WARMUP, measure=base.MEASURE,
                elapsed_s=round(elapsed, 1),
                **res,
            )
            with results.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            flag = "OOM" if res["oom"] else ("FAIL" if res["exit"] != 0 else "ok")
            print(f"[{i}/{len(jobs)}] {tag:<48} {flag:<5} "
                  f"peak={res['peak_mib']:>6}MiB tok/s={res['tokens_per_s']} "
                  f"({elapsed:.0f}s)",
                  flush=True)
        print("done")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
