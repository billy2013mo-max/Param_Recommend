#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按《MFU 实验统计原则》计算 Qwen3-14B 全参 SFT 的多卡 per-step MFU 与 E2E MFU。

与 8B 单卡版（mfu_qwen3_8b/scripts/compute_mfu.py）的三处区别：
  1. 分子换成全参口径（flops_full.step_flops，每个 Linear 6MKN，lm_head 可训练）
  2. N_GPU 由 --gpus 指定（本实验 4）
  3. 分子按**全部 rank 求和**（每张卡各自处理自己的 microbatch，都在算 FLOPs），
     分母取该 step 的 wall-clock。各 rank 的 step time 因 all-reduce 屏障基本同步，
     主结果取**各 rank 的最大值**（真实 wall-clock，不会低估），同时报 rank0 值备查。

口径要点（逐条对应原则文档）：
  §2   P_peak = 989.4e12 FLOP/s/GPU（H800 BF16 dense Tensor Core）
  §3.1 分子 = 逻辑模型 FLOPs，activation recomputation 不计入
  §4.2 稳态主结果 = ΣF / (ΣT · N_GPU · P)，不是逐步 MFU 的算术平均
  §5   E2E 用 launcher 启动前到进程退出的完整 wall-clock
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from flops_full import QWEN3_14B, step_flops

PEAK = 989.4e12  # FLOP/s/GPU，H800 BF16 dense Tensor Core（原则 §2）


def pct(x: float) -> str:
    return f"{x * 100:.3f}%"


def seq_len_dist(lengths: list[int]) -> dict:
    """序列长度分布，用于诊断多卡负载不均衡（统计原则「不采用 Useful-token MFU」一节）。"""
    if not lengths:
        return {"count": 0}
    s = sorted(lengths)

    def q(p: float) -> int:
        if len(s) == 1:
            return s[0]
        pos = p * (len(s) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
        return int(round(s[lo] + (s[hi] - s[lo]) * (pos - lo)))

    return {
        "count": len(s), "min": s[0], "p50": q(0.5), "p90": q(0.9), "max": s[-1],
        "mean": sum(s) / len(s),
    }


def load_ranks(run: Path) -> dict[int, dict[int, dict]]:
    """读全部 clean_timing_rank*.jsonl → {rank: {step: record}}"""
    files = sorted(run.glob("clean_timing_rank*.jsonl"))
    if not files:
        raise SystemExit(f"缺 {run}/clean_timing_rank*.jsonl")
    out: dict[int, dict[int, dict]] = {}
    for f in files:
        rank = int(f.stem.rsplit("rank", 1)[1])
        recs = {}
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[int(r["step"])] = r
        out[rank] = recs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run_mfu_4gpu.sh 的产物目录")
    ap.add_argument("--warmup", type=int, default=5, help="排除的 warmup step 数（原则 §4.2）")
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run_dir)
    n_gpu = args.gpus
    ranks = load_ranks(run)
    if len(ranks) != n_gpu:
        print(f"⚠ 只找到 {len(ranks)} 个 rank 的计时文件，但 --gpus={n_gpu}。"
              f"分子会少算，先查为什么缺 rank。")

    steps = sorted(set.intersection(*[set(v) for v in ranks.values()]))
    if not steps:
        raise SystemExit("没有所有 rank 都记录到的 step")

    per_step = []
    for st in steps:
        lengths: list[int] = []          # 真实长度，供 attention core 的 ΣS²
        padded_lengths: list[int] = []   # padded 长度，Linear 的 GEMM 真算了 padding
        times: dict[int, float] = {}
        mem: list[float] = []
        bad = False
        no_mask = False
        srcs: set[str] = set()   # 真实长度是从哪儿来的：attention_mask / position_ids
        padded_tok = nonpad_tok = label_tok = 0
        # 统计原则「不采用 Useful-token MFU」一节要求的逐 rank 诊断数据
        per_rank_diag: dict[int, dict] = {}
        for rk, recs in sorted(ranks.items()):
            r = recs[st]
            times[rk] = r["optimizer_step_time_s"]
            if r.get("peak_mem_allocated_gib"):
                mem.append(r["peak_mem_allocated_gib"])
            rk_lengths: list[int] = []
            for m in r.get("microbatches") or []:
                if not m or "error" in m:
                    bad = True
                    continue
                if m.get("mask_present") is False:
                    no_mask = True
                srcs.add(m.get("length_source", "?"))
                lengths.extend(m["lengths"])
                rk_lengths.extend(m["lengths"])
                b, s = m["shape"][0], m["shape"][1]
                padded_lengths.extend([s] * b)
            padded_tok += r["padded_tokens"]
            nonpad_tok += r["nonpad_tokens"]
            label_tok += r.get("label_tokens") or 0
            per_rank_diag[rk] = {
                "padded_tokens": r["padded_tokens"],
                "nonpad_tokens": r["nonpad_tokens"],
                "sample_count": len(rk_lengths),
                "seq_length_distribution": seq_len_dist(rk_lengths),
                "step_time_s": r["optimizer_step_time_s"],
            }
        if bad or not lengths:
            print(f"⚠ step {st} shape 记录不完整，跳过（不静默按 0 处理）")
            continue
        if no_mask:
            print(f"⚠ step {st} 既无 attention_mask 也无 position_ids，真实长度按满长度算了，"
                  f"attention core 会偏高 —— 这个数不可用")

        f_lin = step_flops(QWEN3_14B, padded_lengths)
        f_att = step_flops(QWEN3_14B, lengths)
        parts = dict(f_lin)
        parts["attention_core"] = f_att["attention_core"]
        parts["total"] = sum(v for k, v in parts.items() if k != "total")

        t_max = max(times.values())
        # padding 的算力是真花掉了，所以它进分子是对的，不做任何「反事实」扣除。
        # packing / padding / 负载不均衡的诊断改由 data_balance_diagnostics 承担，
        # 统计原则明确禁止把「把 padded token 换成非 padding token 的反事实 FLOPs」
        # 命名为 MFU（见「不采用 Useful-token MFU」一节）。
        per_step.append({
            "step": st,
            "length_source": "+".join(sorted(srcs)) if srcs else "?",
            "step_time_s_max": t_max,
            "step_time_s_rank0": times.get(0, t_max),
            "step_time_s_per_rank": times,
            "padded_tokens": padded_tok,
            "nonpad_tokens": nonpad_tok,
            "label_tokens": label_tok,
            "padding_fraction": 1 - nonpad_tok / padded_tok if padded_tok else 0.0,
            "peak_mem_allocated_gib_max": max(mem) if mem else None,
            "model_flops": parts["total"],
            "components": parts,
            "mfu": parts["total"] / (t_max * n_gpu * PEAK),
            "data_balance_diagnostics": {
                "per_rank": per_rank_diag,
                "padding_fraction": 1 - nonpad_tok / padded_tok if padded_tok else 0.0,
                "global_nonpad_tokens_per_second": nonpad_tok / t_max if t_max else 0.0,
                "straggler_gap_s": t_max - min(times.values()),
                "straggler_gap_percent": ((t_max - min(times.values())) / t_max * 100
                                          if t_max else 0.0),
            },
        })

    warm = args.warmup
    steady = [p for p in per_step if p["step"] > warm]
    if not steady:
        raise SystemExit("稳态窗口为空")

    sum_f = sum(p["model_flops"] for p in steady)
    sum_t = sum(p["step_time_s_max"] for p in steady)
    sum_t0 = sum(p["step_time_s_rank0"] for p in steady)
    steady_mfu = sum_f / (sum_t * n_gpu * PEAK)  # §4.2 主结果
    mfus = sorted(p["mfu"] for p in steady)
    tok = sum(p["padded_tokens"] for p in steady)
    nonpad = sum(p["nonpad_tokens"] for p in steady)
    mems = [p["peak_mem_allocated_gib_max"] for p in steady if p["peak_mem_allocated_gib_max"]]

    def quant(q: float) -> float:
        if len(mfus) == 1:
            return mfus[0]
        pos = q * (len(mfus) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(mfus) - 1)
        return mfus[lo] + (mfus[hi] - mfus[lo]) * (pos - lo)

    summary = {
        "steady_steps": [p["step"] for p in steady],
        "warmup_steps_excluded": warm,
        "step_count": len(steady),
        "total_model_flops": sum_f,
        "summed_step_time_s": sum_t,
        "time_weighted_mfu": steady_mfu,
        "time_weighted_mfu_percent": steady_mfu * 100,
        "time_weighted_mfu_percent_rank0_clock": sum_f / (sum_t0 * n_gpu * PEAK) * 100,
        "mean_step_time_s": sum_t / len(steady),
        "median": statistics.median(mfus),
        "min": mfus[0],
        "max": mfus[-1],
        "p25": quant(0.25),
        "p75": quant(0.75),
        "padded_tokens": tok,
        "nonpad_tokens": nonpad,
        "padding_fraction": 1 - nonpad / tok if tok else 0.0,
        "padded_tokens_per_second": tok / sum_t,
        "nonpad_tokens_per_second": nonpad / sum_t,
        "gflop_per_padded_token": sum_f / tok / 1e9,
        "flops_per_step": steady[0]["model_flops"],
        "peak_mem_allocated_gib_max": max(mems) if mems else None,
        "step_time_cv_percent": (statistics.stdev([p["step_time_s_max"] for p in steady])
                                 / (sum_t / len(steady)) * 100) if len(steady) > 1 else 0.0,
    }

    # ---- data_balance_diagnostics（统计原则「不采用 Useful-token MFU」一节）----
    # 这里放的是 packing / padding / 多卡负载不均衡的诊断量，全部是可观测量，
    # 不构造任何反事实 FLOPs，也不以 MFU 命名。四项：
    #   1) 每个 rank 的 padded / non-padding token 数
    #   2) 每个 rank 的 sequence-length 分布
    #   3) padding fraction 与 global non-padding token/s
    #   4) 每个 rank 的 step 时间与 straggler gap
    rank_ids = sorted(ranks.keys())
    rank_agg = {}
    for rk in rank_ids:
        d = [p["data_balance_diagnostics"]["per_rank"][rk] for p in steady
             if rk in p["data_balance_diagnostics"]["per_rank"]]
        if not d:
            continue
        rt = [x["step_time_s"] for x in d]
        rank_agg[rk] = {
            "padded_tokens": sum(x["padded_tokens"] for x in d),
            "nonpad_tokens": sum(x["nonpad_tokens"] for x in d),
            "sample_count": sum(x["sample_count"] for x in d),
            "seq_length_distribution_last_step": d[-1]["seq_length_distribution"],
            "seq_length_mean_over_window": (
                sum(x["seq_length_distribution"].get("mean", 0) * x["sample_count"] for x in d)
                / max(1, sum(x["sample_count"] for x in d))),
            "summed_step_time_s": sum(rt),
            "mean_step_time_s": sum(rt) / len(rt),
        }
    gaps = [p["data_balance_diagnostics"]["straggler_gap_s"] for p in steady]
    summary["data_balance_diagnostics"] = {
        "per_rank": rank_agg,
        "padding_fraction": summary["padding_fraction"],
        "global_nonpad_tokens_per_second": nonpad / sum_t if sum_t else 0.0,
        "global_padded_tokens_per_second": tok / sum_t if sum_t else 0.0,
        "straggler_gap_s_mean": sum(gaps) / len(gaps),
        "straggler_gap_s_max": max(gaps),
        "straggler_gap_percent_mean": (sum(gaps) / len(gaps)) / (sum_t / len(steady)) * 100,
        "rank_load_imbalance_percent": (
            (max(v["padded_tokens"] for v in rank_agg.values())
             / max(1, min(v["padded_tokens"] for v in rank_agg.values())) - 1) * 100
            if rank_agg else 0.0),
        "note": "诊断用，不得据此计算或命名任何 MFU（统计原则明令禁止 useful-token MFU）",
    }

    report = {
        "mfu_protocol": {
            "version": 1,
            "numerator_mode": "logical_model_flops",
            "trainability": "full_parameter（每个 Linear 6MKN，lm_head 可训练）",
            "recomputation_in_numerator": False,
            "shape_convention": "Linear 用 padded BxS；attention core 用真实序列长度",
            "multi_gpu_numerator": "全部 rank 的 microbatch 求和",
            "multi_gpu_denominator": "各 rank step time 取最大值（真实 wall-clock）",
        },
        "hardware": {
            "gpu_model": "NVIDIA H800", "gpu_count": n_gpu, "precision": "bf16",
            "peak_mode": "dense_tensor_core", "peak_tflops_per_gpu": PEAK / 1e12,
        },
        "per_step_mfu": {"records": per_step, "summary": summary},
    }

    # E2E（原则 §5）
    e2e_end = run / "e2e_end.json"
    if e2e_end.exists():
        wall = json.loads(e2e_end.read_text())["wall_time_s"]
        all_f = sum(p["model_flops"] for p in per_step)
        report["e2e_mfu"] = {
            "timer_start": "launcher_command_start",
            "timer_end": "all_processes_exited",
            "wall_time_s": wall,
            "completed_training_steps": len(per_step),
            "training_model_flops": all_f,
            "evaluation_batches": 0,
            "evaluation_forward_flops": 0,
            "total_logical_model_flops": all_f,
            "final_checkpoint_verified": False,
            "note": "本实验 save_strategy=no、无 eval，故无 checkpoint 持久化项；"
                    "E2E 含初始化/编译/数据加载/收尾，未做任何扣除",
            "mfu": all_f / (wall * n_gpu * PEAK),
            "mfu_percent": all_f / (wall * n_gpu * PEAK) * 100,
        }

    (run / "mfu_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    cfg = run / "effective_config.yaml"
    print(f"\n=== {run.parent.name}/{run.name} ===")
    if cfg.exists():
        import re
        txt = cfg.read_text()
        for key in ("per_device_train_batch_size", "gradient_accumulation_steps",
                    "disable_gradient_checkpointing", "use_unsloth_gc",
                    "cutoff_len", "packing", "enable_cce", "deepspeed"):
            m = re.search(rf"^{key}:\s*(\S+)", txt, re.M)
            if m:
                print(f"  {key:34s} = {m.group(1)}")
    print(f"\n  卡数                   : {n_gpu}（找到 {len(ranks)} 个 rank 的计时）")
    print(f"  稳态窗口               : step {steady[0]['step']}-{steady[-1]['step']}"
          f"（{len(steady)} step，排除前 {warm}）")
    print(f"  逻辑 FLOPs/step        : {summary['flops_per_step']/1e15:.6f} PFLOPs（全 {n_gpu} 卡合计）")
    print(f"  每 padded token        : {summary['gflop_per_padded_token']:.3f} GFLOP")
    print(f"  padding 占比           : {summary['padding_fraction']*100:.1f}%")
    print(f"  稳态累计 FLOPs         : {sum_f/1e15:.6f} PFLOPs")
    print(f"  稳态累计时间           : {sum_t:.6f} s")
    print(f"  平均 step time         : {summary['mean_step_time_s']:.6f} s  (cv {summary['step_time_cv_percent']:.2f}%)")
    print(f"  吞吐(padded)           : {summary['padded_tokens_per_second']:.1f} token/s")
    print(f"  吞吐(非padding)        : {summary['nonpad_tokens_per_second']:.1f} token/s")
    if summary["peak_mem_allocated_gib_max"]:
        cap = 143771 / 1024
        pk = summary["peak_mem_allocated_gib_max"]
        print(f"  显存峰值(最大 rank)    : {pk:.1f} GiB / 安全线 {0.95*cap:.1f} GiB"
              f"（余量 {0.95*cap-pk:.1f}）")
    print(f"  ★ Per-step MFU（时间加权汇总）: {pct(steady_mfu)}")
    print(f"    median={pct(summary['median'])} min={pct(summary['min'])} "
          f"max={pct(summary['max'])} p25={pct(summary['p25'])} p75={pct(summary['p75'])}")
    print(f"    （若用 rank0 时钟做分母: {summary['time_weighted_mfu_percent_rank0_clock']:.3f}%）")
    d = summary["data_balance_diagnostics"]
    print(f"\n  数据均衡诊断（仅诊断，不是 MFU）:")
    print(f"    全局非padding吞吐     : {d['global_nonpad_tokens_per_second']:.1f} token/s")
    print(f"    padding 占比          : {d['padding_fraction']*100:.1f}%")
    print(f"    straggler gap         : 均值 {d['straggler_gap_s_mean']:.4f} s "
          f"({d['straggler_gap_percent_mean']:.2f}% of step)，最大 {d['straggler_gap_s_max']:.4f} s")
    print(f"    rank 间 token 不均衡  : {d['rank_load_imbalance_percent']:.2f}%")
    for rk, v in d["per_rank"].items():
        sd = v["seq_length_distribution_last_step"]
        print(f"    rank{rk}: padded {v['padded_tokens']:>9d}  非padding {v['nonpad_tokens']:>9d}"
              f"  样本 {v['sample_count']:>5d}  长度均值 {v['seq_length_mean_over_window']:.0f}"
              f"  末步 p50/p90/max {sd.get('p50')}/{sd.get('p90')}/{sd.get('max')}"
              f"  平均step {v['mean_step_time_s']:.4f}s")
    if "e2e_mfu" in report:
        e = report["e2e_mfu"]
        print(f"  E2E MFU                : {pct(e['mfu'])}  (wall {e['wall_time_s']:.3f}s)")
    print(f"\n  报告: {run/'mfu_report.json'}")


if __name__ == "__main__":
    main()
