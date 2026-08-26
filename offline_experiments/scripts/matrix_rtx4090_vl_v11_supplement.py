#!/usr/bin/env python3
"""4090 VL v1 补数矩阵：填多卡 gc=on 与单卡 gc=off 两个吞吐失败模式的空白。

设计依据（来自 3c out-of-sample 揭示的结构性偏差）：
  - 多卡 gc=on：拟合池原本 0 覆盖，OOS 补了 5 组，仍嫌少；本轮补齐 3B 全网格
  - 单卡 gc=off：OOS 只有 2 组（都 mbs=1），本轮补 mbs=2 覆盖

复用原矩阵执行框架。结果写入独立目录，最后合并到拟合池。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path("/wanqing-develop/luowenjing/Param_Recommend/offline_experiments"
            "/artifacts/rtx4090_vl_observations_20260823")
sys.path.insert(0, str(_SRC))
import matrix as base  # type: ignore
import yaml  # noqa: E402


def build_jobs():
    jobs = []
    # 多卡 gc=on 全网格（3B）：18 组
    for ds in ("blind_f1", "blind_f2", "blind_f4"):
        for mbs in (1, 2, 4):
            for gpu in (2, 4):
                jobs.append(dict(
                    model="qwen25vl3b", dataset=ds, mbs=mbs,
                    gpu_count=gpu, gc=True, zero="z2",
                    note=f"3b_multi_gc_on_{ds}_mbs{mbs}_g{gpu}",
                ))
    # 单卡 gc=off 补 mbs≥2 的空白：4 组
    for model, ds, mbs in [
        ("qwen25vl3b", "blind_f1", 2),
        ("qwen25vl3b", "blind_f2", 1),
        ("qwen3vl4b",  "blind_f1", 2),
        ("qwen3vl4b",  "blind_f2", 1),
    ]:
        jobs.append(dict(
            model=model, dataset=ds, mbs=mbs,
            gpu_count=1, gc=False, zero="none",
            note=f"single_gc_off_{model.split('vl')[0]}b_{ds}_mbs{mbs}",
        ))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/vl4090_supp_v11"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.is_file():
        done = {json.loads(l)["tag"] for l in results.open()}

    jobs = build_jobs()
    print(f"计划 {len(jobs)} 个补数作业，已完成 {len(done)} 个")
    if args.dry_run:
        for j in jobs: print(" ", j)
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
                out_dir=out_dir)
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
                **res)
            with results.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            flag = "OOM" if res["oom"] else ("FAIL" if res["exit"] != 0 else "ok")
            print(f"[{i}/{len(jobs)}] {tag:<48} {flag:<5} "
                  f"peak={res['peak_mib']:>6}MiB tok/s={res['tokens_per_s']} "
                  f"({elapsed:.0f}s)", flush=True)
        print("done")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
