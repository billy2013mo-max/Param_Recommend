#!/usr/bin/env python3
"""4090 上 Qwen3.5 的 packing 矩阵：attention × packing 模式 × cutoff_len。

为什么要这一轮（前 472 个观测里 packing 全程是关的）：
  - llamafactory 给 Qwen3.5 挂的 patcher 本来就是为变长序列拼接而写的；
    它把 position_ids 转成 cu_seqlens 交给 fla，这正是 packing 的形态。
  - 实测单点：4B/text/short_512/单卡，FA2 无 packing 292 tok/s，
    FA2+packing 475，FA2+neat_packing 794（2.7 倍）。
  - 因此「sdpa+大 MBS 是 4090 最优」这个结论只在不开 packing 时成立。

关键设计点 cutoff_len：
  不开 packing 时它不驱动显存（VL 矩阵实测 4096 与 8192 峰值常常完全相同）；
  一旦开 packing，它直接决定拼接后每条序列的长度，会重新成为主要驱动因素。
  所以这一轮必须把它扫回来。

约束（均已实测）：
  - neat_packing 要求 batch=1（否则 AssertionError: bsz should be 1）
  - FA2 路径下 MBS 也只能是 1（patcher + fla 的 cu_seqlens 限制）
  - 所以 packing 与大 MBS 是替代关系，不能叠加
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import yaml

VENV = "/wanqing-develop/luowenjing/Param_Recommend/qwen36_venv/bin/python"
TORCHRUN = "/wanqing-develop/luowenjing/Param_Recommend/qwen36_venv/bin/torchrun"
REPO = Path("/wanqing-develop/luowenjing/Param_Recommend/offline_experiments")
DS_DIR = REPO / "campaigns/rtx4090_20260717/config/deepspeed"
DATA_ROOT = REPO / "data"

MODELS = {
    "qwen35_0p8b": dict(path="/wanqing-models/Qwen3.5-0.8B", single_card_ok=True),
    "qwen35_4b": dict(path="/wanqing-models/Qwen3.5-4B", single_card_ok=True),
    "qwen35_9b": dict(path="/wanqing-models/Qwen3.5-9B", single_card_ok=False),
}

# packing 的收益来自填满序列，所以短样本数据集最能体现差异；
# 长序列数据集用来看 packing 是否反而带来浪费。
TEXT_DATASETS = {
    "short_512": dict(name="short_512", dir=DATA_ROOT),
    "multiturn_4096": dict(name="multiturn_4096", dir=DATA_ROOT),
}

TEMPLATE = "qwen3_5_nothink"
WARMUP, MEASURE = 2, 8
LOCK = Path("/tmp/rtx4090_matrix.lock")


def build_cfg(*, model, dataset_key, attn, pack_mode, cutoff, mbs, gpu_count,
              gc, zero, out_dir):
    m, d = MODELS[model], TEXT_DATASETS[dataset_key]
    packing = pack_mode in ("pack", "neat")
    cfg = {
        "model_name_or_path": m["path"],
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "dataset": d["name"],
        "dataset_dir": str(d["dir"]),
        "template": TEMPLATE,
        "cutoff_len": cutoff,
        "max_samples": 1000,
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 0,
        "overwrite_cache": False,
        "packing": packing,
        "neat_packing": pack_mode == "neat",
        "output_dir": str(out_dir),
        "overwrite_output_dir": True,
        "logging_steps": 1,
        "logging_strategy": "steps",
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
        "plot_loss": False,
        "disable_tqdm": True,
        "per_device_train_batch_size": mbs,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.0e-05,
        "max_steps": WARMUP + MEASURE,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "flash_attn": attn,
        "enable_liger_kernel": True,
        "optim": "adamw_torch_fused",
        "torch_compile": False,
        "gradient_checkpointing": gc,
        "disable_gradient_checkpointing": not gc,
        "use_reentrant_gc": True,
        "include_num_input_tokens_seen": "all",
        "seed": 20260716,
        "data_seed": 20260716,
        "ddp_timeout": 180000000,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target": "all",
    }
    if zero in ("z2", "z3"):
        cfg["deepspeed"] = str(DS_DIR / f"ds_{zero}.json")
    return cfg


def _baseline_mib(devs):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", devs],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return max(int(x) for x in out.splitlines() if x.strip())
    except Exception:
        return 0


def sample_peak(devs, stop, peak):
    """按卡采样并扣除启动前基线。互斥锁保证同一时间只有本矩阵占卡。

    不用 pid 匹配：nvidia-smi 报宿主命名空间 pid，容器内 ps 是另一套编号，
    父子树永远对不上（已踩过，导致峰值全记 0）。
    """
    baseline = _baseline_mib(devs)
    peak[1] = baseline
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", devs],
                capture_output=True, text=True, timeout=10).stdout.strip()
            vals = [int(x) for x in out.splitlines() if x.strip()]
            if vals:
                peak[0] = max(peak[0], max(vals) - baseline)
        except Exception:
            pass
        time.sleep(0.25)


def run_one(yaml_path, gpu_count, cwd):
    devs = ",".join(str(i) for i in range(gpu_count))
    peak = [0, 0]
    stop = threading.Event()
    t = threading.Thread(target=sample_peak, args=(devs, stop, peak), daemon=True)
    t.start()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = devs
    if gpu_count == 1:
        cmd = [VENV, "-m", "llamafactory.launcher", str(yaml_path)]
    else:
        cmd = [TORCHRUN, "--nnodes", "1", "--nproc_per_node", str(gpu_count),
               "--master_port", "29583", "-m", "llamafactory.launcher",
               str(yaml_path)]
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env=env, cwd=str(cwd))
        log, _ = proc.communicate(timeout=3600)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
        log, code = "TIMEOUT", -9
    stop.set()
    t.join(timeout=2)
    oom = any(s in log for s in ("OutOfMemoryError", "CUDA out of memory",
                                 "Triton Error [CUDA]: out of memory",
                                 "CUDA error: out of memory"))
    tps = re.findall(r"train_tokens_per_second[^0-9]*([0-9.]+)", log)
    sps = re.findall(r"train_samples_per_second\s*=\s*([0-9.]+)", log)
    tok = re.findall(r"num_input_tokens_seen\s*=\s*(\d+)", log)
    return dict(exit=code, oom=oom, peak_mib=peak[0], baseline_mib=peak[1],
                tokens_per_s=float(tps[-1]) if tps else None,
                samples_per_s=float(sps[-1]) if sps else None,
                tokens_seen=int(tok[-1]) if tok else None,
                log_tail=log[-1500:] if (code != 0 and not oom) else None)


def plan():
    """多卡轨：按单卡 60 组实测结果裁剪，只跑确定有意义的组合。

    单卡实测结论（tok/s）：
      - FA2 + packing 是最优路径，短样本上比 FA2 基线快最高 20 倍
        （0.8B/short_512: 197 → 3698）。packing 消掉了 padding 浪费。
      - cutoff_len 开 packing 后成为主要驱动：0.8B/multiturn
        c2048→c8192 吞吐 3638 → 9444。
      - sdpa + packing 全部失败（12/12），报掩码尺寸不匹配
        （expanded size 2120 vs existing 2192）——sdpa 的注意力掩码
        与 packing 的变长序列在此版本不兼容。两条优化路线互斥。
      - 4B 上 packing 在 c4096 及以上全 OOM，只有 c2048 可行。

    据此裁掉：sdpa+neat（必失败）、4B/9B 的 c4096+ packing（必 OOM）。
    """
    jobs = []
    for model, m in MODELS.items():
        for ds_key in TEXT_DATASETS:
            for cutoff in (2048, 4096, 8192):
                for attn, pack_mode, mbs in (
                    ("fa2", "off", 1),
                    ("fa2", "pack", 1),
                    ("fa2", "neat", 1),
                    ("sdpa", "off", 4),
                ):
                    # sdpa+packing 组合已实测必然失败，不再占机时
                    if attn == "sdpa" and pack_mode in ("pack", "neat"):
                        continue
                    # 4B/9B 的 packing 在 c4096 及以上单卡已 OOM；多卡分片
                    # 只切分优化器状态，激活仍按卡算，故同样跳过
                    if (model in ("qwen35_4b", "qwen35_9b")
                            and pack_mode in ("pack", "neat")
                            and cutoff >= 4096):
                        continue
                    for gpu_count in (2, 4):
                        zeros = ("z2", "z3") if model == "qwen35_9b" else ("z2",)
                        for zero in zeros:
                            jobs.append(dict(model=model, dataset=ds_key,
                                             attn=attn, pack_mode=pack_mode,
                                             cutoff=cutoff, mbs=mbs,
                                             gpu_count=gpu_count, gc=False,
                                             zero=zero))
    return jobs


def acquire_lock():
    if LOCK.exists():
        pid = -1
        try:
            holder = json.loads(LOCK.read_text())
            pid = int(holder.get("pid", -1))
            os.kill(pid, 0)
        except (ProcessLookupError, ValueError, json.JSONDecodeError):
            print(f"发现过期锁（持有者已退出），接管 {LOCK}")
        except PermissionError:
            raise SystemExit(f"另一个矩阵正在运行（pid={pid}），拒绝并发占卡")
        else:
            raise SystemExit(
                f"另一个矩阵正在运行：pid={pid}\n"
                f"并发会污染显存采样并制造假 OOM。等它跑完，或删除 {LOCK}")
    LOCK.write_text(json.dumps({"pid": os.getpid(), "what": "packing_matrix",
                                "started": time.strftime("%F %T")}))
    return LOCK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/qwen35_pack/matrix"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.is_file():
        done = {json.loads(l)["tag"] for l in results.open()}

    tagged = []
    for j in plan():
        tag = (f"{j['model']}_{j['dataset']}_{j['attn']}_{j['pack_mode']}"
               f"_c{j['cutoff']}_mbs{j['mbs']}_g{j['gpu_count']}_{j['zero']}")
        if args.filter and args.filter not in tag:
            continue
        tagged.append((tag, j))
    if args.limit:
        tagged = tagged[: args.limit]
    print(f"计划 {len(tagged)} 个作业，已完成 {len(done)} 个")
    if args.dry_run:
        for tag, _ in tagged:
            print("  ", tag)
        return

    lock = acquire_lock()
    try:
        for i, (tag, j) in enumerate(tagged, 1):
            if tag in done:
                continue
            out_dir = args.out / f"out_{tag}"
            cfg = build_cfg(model=j["model"], dataset_key=j["dataset"],
                            attn=j["attn"], pack_mode=j["pack_mode"],
                            cutoff=j["cutoff"], mbs=j["mbs"],
                            gpu_count=j["gpu_count"], gc=j["gc"],
                            zero=j["zero"], out_dir=out_dir)
            yaml_path = args.out / f"cfg_{tag}.yaml"
            yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True,
                                                sort_keys=False))
            res = run_one(yaml_path, j["gpu_count"], args.out)
            row = dict(tag=tag, **j, warmup=WARMUP, measure=MEASURE, **res)
            with results.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            flag = "OOM" if res["oom"] else ("FAIL" if res["exit"] != 0 else "ok")
            print(f"[{i}/{len(tagged)}] {tag:<58} {flag:<5} "
                  f"peak={res['peak_mib']:>6}MiB tok/s={res['tokens_per_s']}",
                  flush=True)
    finally:
        lock.unlink(missing_ok=True)
    print("done")


if __name__ == "__main__":
    main()
