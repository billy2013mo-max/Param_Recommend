#!/usr/bin/env python3
"""4090 VL 显存/吞吐边界探测：只写 /tmp，不碰仓库产物。"""
import json, os, re, subprocess, sys, threading, time
from pathlib import Path

VENV = "/fine-tuning-launcher/.venv/bin/python"
BASE = Path("/tmp/vl4090_smoke")
OUT = BASE / "sweep_results.jsonl"

MODELS = {
    "qwen25vl3b": dict(base="smoke_qwen25vl3b.yaml"),
    "qwen3vl4b": dict(base="smoke_qwen3vl4b.yaml"),
}


def sample_peak(dev, stop, peak):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-i", dev],
                capture_output=True, text=True, timeout=10).stdout.strip()
            peak[0] = max(peak[0], int(out.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.25)


def run_one(cfg_path, dev):
    peak = [0]
    stop = threading.Event()
    t = threading.Thread(target=sample_peak, args=(dev, stop, peak), daemon=True)
    t.start()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = dev
    try:
        r = subprocess.run([VENV, "-m", "llamafactory.launcher", str(cfg_path)],
                           capture_output=True, text=True, env=env, timeout=1800)
        code, log = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        code, log = -9, "TIMEOUT"
    stop.set()
    t.join(timeout=2)
    oom = "OutOfMemoryError" in log or "CUDA out of memory" in log
    tps = re.findall(r"train_tokens_per_second[^0-9]*([0-9.]+)", log)
    sps = re.findall(r"train_samples_per_second\s*=\s*([0-9.]+)", log)
    rt = re.findall(r"train_runtime\s*=\s*([0-9:.]+)", log)
    return dict(exit=code, oom=oom, peak_mib=peak[0],
                tokens_per_s=float(tps[-1]) if tps else None,
                samples_per_s=float(sps[-1]) if sps else None,
                train_runtime=rt[-1] if rt else None)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "3"
    rows = []
    for model, spec in MODELS.items():
        base = (BASE / spec["base"]).read_text()
        for frames in (1, 2, 4):
            for mbs in (1, 2, 4):
                for cutoff in (4096, 8192):
                    tag = f"{model}_f{frames}_mbs{mbs}_c{cutoff}"
                    out_dir = BASE / f"out_{tag}"
                    cfg = base
                    cfg = re.sub(r"per_device_train_batch_size: \d+",
                                 f"per_device_train_batch_size: {mbs}", cfg)
                    cfg = re.sub(r"/f\d+/registry", f"/f{frames}/registry", cfg)
                    cfg = re.sub(r"output_dir: .*", f"output_dir: {out_dir}", cfg)
                    cfg = re.sub(r"cutoff_len: \d+", f"cutoff_len: {cutoff}", cfg)
                    cfg_path = BASE / f"cfg_{tag}.yaml"
                    cfg_path.write_text(cfg)
                    res = run_one(cfg_path, dev)
                    row = dict(tag=tag, model=model, frames=frames, mbs=mbs,
                               cutoff_len=cutoff, **res)
                    rows.append(row)
                    with OUT.open("a") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"{tag:<34} oom={str(res['oom']):<5} "
                          f"peak={res['peak_mib']:>6}MiB tok/s={res['tokens_per_s']}",
                          flush=True)
    print(f"\ndone: {len(rows)} runs -> {OUT}")


if __name__ == "__main__":
    main()
