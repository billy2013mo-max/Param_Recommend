#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按《MFU 实验统计原则》计算 per-step MFU 与 E2E MFU。

输入：run_mfu.sh 产出的 clean_timing.jsonl（含逐 step 时间与真实 shape）
输出：mfu_report.json + 终端表格

口径要点（逐条对应原则文档）：
  §2   P_peak = 989.4e12 FLOP/s（H800 BF16 dense Tensor Core），N_GPU = 1
  §3.1 分子 = 逻辑模型 FLOPs，activation recomputation 不计入
  §3.2 LoRA：冻结 base Linear 用 4MKN，adapter 用 6Mr(K+N)
  §4.2 稳态主结果 = ΣF / (ΣT · N · P)，不是逐步 MFU 的算术平均
  §5   E2E 用 launcher 启动前到进程退出的完整 wall-clock
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from flops_lora import ALL_TARGETS, QWEN3_8B, step_flops

PEAK = 989.4e12  # FLOP/s/GPU，H800 BF16 dense Tensor Core（原则 §2）
N_GPU = 1


def pct(x: float) -> str:
    return f"{x * 100:.3f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run_mfu.sh 的产物目录")
    ap.add_argument("--warmup", type=int, default=5, help="排除的 warmup step 数（原则 §4.2）")
    ap.add_argument("--lora-rank", type=int, default=32)
    args = ap.parse_args()

    run = Path(args.run_dir)
    timing = run / "clean_timing.jsonl"
    if not timing.exists():
        raise SystemExit(f"缺 {timing}")

    records = [json.loads(l) for l in timing.read_text().splitlines() if l.strip()]
    if not records:
        raise SystemExit("clean_timing.jsonl 为空")

    # 修复前采集的 clean_timing.jsonl 没有 segmentation 字段。只有 neat_packing 的
    # run 受影响：它用块对角掩码让每条样本只看自己，attention 必须按逐样本算，而当时
    # 记的是整包总长（collator 把 mask 置 None，钩子走了整行 fallback），虚高约
    # "每包样本数"倍。
    #
    # 普通 packing（packing=true / neat_packing=false）不受影响：那种模式下同一包内
    # 的样本**确实互相 attend**，整包就是一个真实的注意力块，整包长度才是对的口径；
    # 且它的 attention_mask 是全 1，当时的 mask.sum() 正好得到整包非 pad 长度。
    neat_packing_on = False
    cfg_path = run / "effective_config.yaml"
    if cfg_path.exists():
        neat_packing_on = bool(
            # 行尾常带 `# 注释`，所以只锚定 key 和值，不要求值后立刻换行。
            re.search(r"^neat_packing:\s*true\b", cfg_path.read_text(), re.M | re.I)
        )
    legacy = [
        m
        for r in records
        for m in (r.get("microbatches") or [])
        if m and "error" not in m and "segmentation" not in m
    ]
    if legacy and neat_packing_on:
        raise SystemExit(
            f"{timing} 是分段口径修复前采集的（{len(legacy)} 个 microbatch 缺 "
            "segmentation 字段），且该 run 开了 neat_packing。块对角 attention 的分子"
            "在这些记录里按整包平方计算，会虚高约每包样本数倍，必须用修复后的钩子"
            "重新采集，不能复算。"
        )
    if legacy:
        print(
            f"注意：{len(legacy)} 个 microbatch 缺 segmentation 字段（修复前采集）。"
            "该 run 未开 neat_packing，同包内样本互相 attend，整包长度即正确口径，"
            "结果不受分段修复影响。"
        )

    per_step = []
    for r in records:
        # 收集该 step 全部 microbatch 的真实分段长度，供 attention core 的 ΣS² 使用。
        # neat_packing 下一行含多条样本，lengths 是逐样本的，不是整行的（块对角掩码
        # 让每条只看自己）；用整行长度会把 attention 分子放大约"每包样本数"倍。
        lengths: list[int] = []
        padded_lengths: list[int] = []
        bad = False
        unsegmented = False
        for m in r.get("microbatches") or []:
            if not m or "error" in m:
                bad = True
                continue
            if m.get("segmentation") == "unsegmented_row_fallback":
                unsegmented = True
            lengths.extend(m["lengths"])
            b, s = m["shape"][0], m["shape"][1]
            padded_lengths.extend([s] * b)
        if bad or not lengths:
            print(f"⚠ step {r['step']} shape 记录不完整，跳过（不静默按 0 处理）")
            continue
        if unsegmented:
            # 拿不到分段就无法算对 attention。宁可跳过也不要报一个虚高的 MFU。
            print(
                f"⚠ step {r['step']} 缺分段信息（segmentation=unsegmented_row_fallback），"
                "跳过：packing 下整行长度会高估 attention 分子"
            )
            continue

        # shape 口径：Linear 用 padded BxS，attention 用真实长度（与参考实验一致）
        f_lin = step_flops(QWEN3_8B, padded_lengths, args.lora_rank, ALL_TARGETS)
        f_att = step_flops(QWEN3_8B, lengths, args.lora_rank, ALL_TARGETS)
        parts = dict(f_lin)
        parts["attention_core"] = f_att["attention_core"]
        parts["total"] = sum(v for k, v in parts.items() if k != "total")

        t = r["optimizer_step_time_s"]
        per_step.append({
            "step": r["step"],
            "optimizer_step_time_s": t,
            "padded_tokens": r["padded_tokens"],
            "nonpad_tokens": r["nonpad_tokens"],
            "label_tokens": r.get("label_tokens"),
            "padding_fraction": 1 - r["nonpad_tokens"] / r["padded_tokens"]
            if r["padded_tokens"] else 0.0,
            "model_flops": parts["total"],
            "components": parts,
            "mfu": parts["total"] / (t * N_GPU * PEAK),
        })

    warm = args.warmup
    steady = [p for p in per_step if p["step"] > warm]
    if not steady:
        raise SystemExit("稳态窗口为空")

    sum_f = sum(p["model_flops"] for p in steady)
    sum_t = sum(p["optimizer_step_time_s"] for p in steady)
    steady_mfu = sum_f / (sum_t * N_GPU * PEAK)  # §4.2 主结果
    mfus = sorted(p["mfu"] for p in steady)
    tok = sum(p["padded_tokens"] for p in steady)

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
        "mean_step_time_s": sum_t / len(steady),
        "median": statistics.median(mfus),
        "min": mfus[0],
        "max": mfus[-1],
        "p25": quant(0.25),
        "p75": quant(0.75),
        "padded_tokens": tok,
        "padded_tokens_per_second": tok / sum_t,
        "gflop_per_padded_token": sum_f / tok / 1e9,
        "flops_per_step": steady[0]["model_flops"],
    }

    report = {
        "mfu_protocol": {
            "version": 1,
            "numerator_mode": "logical_model_flops",
            "recomputation_in_numerator": False,
            "shape_convention": "Linear 用 padded BxS；attention core 用真实序列长度",
        },
        "hardware": {
            "gpu_model": "NVIDIA H800", "gpu_count": N_GPU, "precision": "bf16",
            "peak_mode": "dense_tensor_core", "peak_tflops_per_gpu": PEAK / 1e12,
        },
        "per_step_mfu": {"records": per_step, "summary": summary},
    }

    # E2E（原则 §5）：只有拿到 launcher 边界的 T_start/T_end 才填
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
            "mfu": all_f / (wall * N_GPU * PEAK),
            "mfu_percent": all_f / (wall * N_GPU * PEAK) * 100,
        }

    (run / "mfu_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    cfg = (run / "effective_config.yaml")
    print(f"\n=== {run.name} ===")
    if cfg.exists():
        txt = cfg.read_text()
        for key in ("per_device_train_batch_size", "gradient_accumulation_steps",
                    "use_unsloth_gc", "lora_dropout", "packing", "cutoff_len"):
            m = re.search(rf"^{key}:\s*(\S+)", txt, re.M)
            if m:
                print(f"  {key:32s} = {m.group(1)}")
    print(f"\n  稳态窗口: step {steady[0]['step']}-{steady[-1]['step']}（{len(steady)} step，排除前 {warm}）")
    print(f"  逻辑 FLOPs/step        : {summary['flops_per_step']} = {summary['flops_per_step']/1e15:.6f} PFLOPs")
    print(f"  每 token 逻辑 FLOPs    : {summary['gflop_per_padded_token']:.9f} GFLOP/token")
    print(f"  稳态累计 FLOPs         : {sum_f/1e15:.6f} PFLOPs")
    print(f"  稳态累计时间           : {sum_t:.6f} s")
    print(f"  平均 step time         : {summary['mean_step_time_s']:.6f} s")
    print(f"  吞吐                   : {summary['padded_tokens_per_second']:.3f} token/s")
    print(f"  ★ Per-step MFU（时间加权汇总）: {pct(steady_mfu)}")
    print(f"    median={pct(summary['median'])} min={pct(summary['min'])} "
          f"max={pct(summary['max'])} p25={pct(summary['p25'])} p75={pct(summary['p75'])}")
    if "e2e_mfu" in report:
        e = report["e2e_mfu"]
        print(f"  E2E MFU                : {pct(e['mfu'])}  (wall {e['wall_time_s']:.3f}s)")
    print(f"\n  报告: {run/'mfu_report.json'}")


if __name__ == "__main__":
    main()
