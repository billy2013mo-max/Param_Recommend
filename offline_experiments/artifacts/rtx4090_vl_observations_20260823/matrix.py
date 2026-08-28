#!/usr/bin/env python3
"""4090 VL 正式观测矩阵：单卡与多卡、GC、ZeRO、跨数据源。

设计依据（来自 36 组单卡扫描的实测结论）：
  - cutoff_len 不驱动峰值显存，固定 4096，不作为变量
  - 显存由每样本图数 f 与 MBS 主导，作为主扫描轴
  - 多卡下 ZeRO 2/3 与 GC 是新变量，单卡不适用 ZeRO
只写 /tmp 与显式 --out 指定的目录，不改仓库既有产物。
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

VENV = "/fine-tuning-launcher/.venv/bin/python"
DS_DIR = Path("/wanqing-develop/luowenjing/Param_Recommend/offline_experiments"
              "/campaigns/rtx4090_20260717/config/deepspeed")
DATA_ROOT = Path("/wanqing-develop/luowenjing/Param_Recommend/offline_experiments/data")

MODELS = {
    "qwen25vl3b": dict(
        path="/wanqing-models/Qwen2.5-VL-3B-Instruct",
        template="qwen2_vl", liger=True, min_pixels=3136,
    ),
    "qwen3vl4b": dict(
        path="/wanqing-models/Qwen3-VL-4B-Instruct",
        template="qwen3_vl", liger=False, min_pixels=4096,
    ),
}

# 数据源：盲测集三个分层 + 标定集（固定每样本 2 图，跨来源对照）
DATASETS = {
    "blind_f1": dict(name="vl_blind_prospective_v1",
                     dir=DATA_ROOT / "blind_vl_prospective_v1/f1/registry", frames=1),
    "blind_f2": dict(name="vl_blind_prospective_v1",
                     dir=DATA_ROOT / "blind_vl_prospective_v1/f2/registry", frames=2),
    "blind_f4": dict(name="vl_blind_prospective_v1",
                     dir=DATA_ROOT / "blind_vl_prospective_v1/f4/registry", frames=4),
    "calib_f2": dict(name="vl_pzfj38_calibration_v1", dir=DATA_ROOT, frames=2),
}

CUTOFF = 4096
WARMUP, MEASURE = 2, 8


def build_cfg(*, model, dataset, mbs, gpu_count, gc, zero, out_dir):
    m, d = MODELS[model], DATASETS[dataset]
    cfg = {
        "model_name_or_path": m["path"],
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "dataset": d["name"],
        "dataset_dir": str(d["dir"]),
        "template": m["template"],
        "cutoff_len": CUTOFF,
        "max_samples": 1000,
        "preprocessing_num_workers": 8,
        "dataloader_num_workers": 0,
        "overwrite_cache": False,
        "packing": False,
        "neat_packing": False,
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
        "flash_attn": "fa2",
        "enable_liger_kernel": m["liger"],
        "optim": "adamw_torch_fused",
        "torch_compile": False,
        "gradient_checkpointing": gc,
        "disable_gradient_checkpointing": not gc,
        "use_reentrant_gc": True,
        "include_num_input_tokens_seen": "all",
        "seed": 20260716,
        "data_seed": 20260716,
        "ddp_timeout": 180000000,
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "image_min_pixels": m["min_pixels"],
        "image_max_pixels": 589824,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target": "all",
    }
    if zero in ("z2", "z3"):
        cfg["deepspeed"] = str(DS_DIR / f"ds_{zero}.json")
    return cfg


def _preexisting_gpu_mib(devs):
    """记录作业启动前这些卡上的显存占用，作为基线扣除。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", devs],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return max(int(x) for x in out.splitlines() if x.strip())
    except Exception:
        return 0


def sample_peak(devs, stop, peak, proc_holder):
    """按卡采样峰值，扣除启动前基线。

    不用 --query-compute-apps 的 pid 匹配：nvidia-smi 报的是宿主命名空间 pid，
    容器内 ps 看到的是另一套，父子树永远对不上（已实测）。
    改为「独占 + 基线扣除」：互斥锁保证同一时间只有本矩阵占卡，
    启动前基线扣掉其它常驻进程，剩下的增量就是本作业的用量。
    """
    baseline = _preexisting_gpu_mib(devs)
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


def run_one(cfg_path, gpu_count, base_dir):
    devs = ",".join(str(i) for i in range(gpu_count))
    peak = [0, 0]  # [峰值增量, 启动前基线]
    stop = threading.Event()
    holder: dict = {"proc": None}
    t = threading.Thread(target=sample_peak,
                         args=(devs, stop, peak, holder), daemon=True)
    t.start()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = devs
    if gpu_count == 1:
        cmd = [VENV, "-m", "llamafactory.launcher", str(cfg_path)]
    else:
        cmd = ["/fine-tuning-launcher/.venv/bin/torchrun",
               "--nnodes", "1", "--nproc_per_node", str(gpu_count),
               "--master_port", "29571",
               "-m", "llamafactory.launcher", str(cfg_path)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env=env, cwd=str(base_dir))
        holder["proc"] = proc
        log, _ = proc.communicate(timeout=3600)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        log, code = "TIMEOUT", -9
    stop.set()
    t.join(timeout=2)
    oom = "OutOfMemoryError" in log or "CUDA out of memory" in log
    tps = re.findall(r"train_tokens_per_second[^0-9]*([0-9.]+)", log)
    sps = re.findall(r"train_samples_per_second\s*=\s*([0-9.]+)", log)
    tok = re.findall(r"num_input_tokens_seen\s*=\s*(\d+)", log)
    return dict(exit=code, oom=oom, peak_mib=peak[0],
                baseline_mib=peak[1],
                tokens_per_s=float(tps[-1]) if tps else None,
                samples_per_s=float(sps[-1]) if sps else None,
                tokens_seen=int(tok[-1]) if tok else None,
                log_tail=log[-1200:] if (code != 0 and not oom) else None)


def plan():
    """按 f×MBS 张成，多卡加 ZeRO/GC，跳过单卡扫描已确认必 OOM 的点。"""
    # 单卡已知 OOM 边界（来自 36 组扫描），多卡不外推、仍然实测
    single_oom = {
        ("qwen25vl3b", 2, 4), ("qwen25vl3b", 4, 2), ("qwen25vl3b", 4, 4),
        ("qwen3vl4b", 1, 4), ("qwen3vl4b", 2, 2), ("qwen3vl4b", 2, 4),
        ("qwen3vl4b", 4, 1), ("qwen3vl4b", 4, 2), ("qwen3vl4b", 4, 4),
    }
    jobs = []
    for model in MODELS:
        for ds_key, ds in DATASETS.items():
            f = ds["frames"]
            for mbs in (1, 2, 4):
                # 单卡：只补 GC 开关这一新维度（无 GC 的已在扫描里）
                if (model, f, mbs) not in single_oom:
                    jobs.append(dict(model=model, dataset=ds_key, mbs=mbs,
                                     gpu_count=1, gc=True, zero="none"))
                # 多卡：2/4 卡 × ZeRO2/3，GC 关；显存翻倍可能救回单卡 OOM 点
                for gpu_count in (2, 4):
                    for zero in ("z2", "z3"):
                        jobs.append(dict(model=model, dataset=ds_key, mbs=mbs,
                                         gpu_count=gpu_count, gc=False, zero=zero))
    return jobs


LOCK = Path("/tmp/rtx4090_matrix.lock")


def acquire_lock():
    """全局互斥：同一时间只允许一个 4090 矩阵占卡。

    并发跑两个矩阵会让显存采样互相污染，并且制造假 OOM。
    """
    if LOCK.exists():
        try:
            holder = json.loads(LOCK.read_text())
            pid = int(holder.get("pid", -1))
            os.kill(pid, 0)  # 探活，不发真信号
        except (ProcessLookupError, ValueError, json.JSONDecodeError):
            print(f"发现过期锁（持有者已退出），接管 {LOCK}")
        except PermissionError:
            raise SystemExit(f"另一个矩阵正在运行（pid={pid}），拒绝并发占卡")
        else:
            raise SystemExit(
                f"另一个矩阵正在运行：{holder}\n"
                f"并发会污染显存采样并制造假 OOM。等它跑完，或删除 {LOCK}")
    LOCK.write_text(json.dumps({"pid": os.getpid(), "what": "vl_matrix",
                                "started": time.strftime("%F %T")}))
    return LOCK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/vl4090_matrix"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.is_file():
        done = {json.loads(l)["tag"] for l in results.open()}

    jobs = plan()
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"计划 {len(jobs)} 个作业，已完成 {len(done)} 个")
    if args.dry_run:
        for j in jobs:
            print("  ", j)
        return

    lock = acquire_lock()
    try:
        run_all(jobs, done, args, results)
    finally:
        lock.unlink(missing_ok=True)


def run_all(jobs, done, args, results):
    for i, j in enumerate(jobs, 1):
        tag = (f"{j['model']}_{j['dataset']}_mbs{j['mbs']}"
               f"_g{j['gpu_count']}_gc{int(j['gc'])}_{j['zero']}")
        if tag in done:
            continue
        out_dir = args.out / f"out_{tag}"
        cfg = build_cfg(model=j["model"], dataset=j["dataset"], mbs=j["mbs"],
                        gpu_count=j["gpu_count"], gc=j["gc"], zero=j["zero"],
                        out_dir=out_dir)
        yaml_path = args.out / f"cfg_{tag}.yaml"
        yaml_path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        res = run_one(yaml_path, j["gpu_count"], args.out)
        row = dict(tag=tag, **j, frames=DATASETS[j["dataset"]]["frames"],
                   cutoff_len=CUTOFF, warmup=WARMUP, measure=MEASURE, **res)
        with results.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        flag = "OOM" if res["oom"] else ("FAIL" if res["exit"] != 0 else "ok")
        print(f"[{i}/{len(jobs)}] {tag:<46} {flag:<5} "
              f"peak={res['peak_mib']:>6}MiB tok/s={res['tokens_per_s']}",
              flush=True)
    print("done")


if __name__ == "__main__":
    main()
