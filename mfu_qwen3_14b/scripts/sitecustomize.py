# -*- coding: utf-8 -*-
"""合并委托 sitecustomize：venv 钩子 + per-step MFU 计时钩子（多卡版）。

与 8B 单卡版（mfu_qwen3_8b/scripts/sitecustomize.py）的唯一区别：
  **每个 rank 各写自己的计时文件**，文件名后缀 _rank{LOCAL_RANK}。
  单卡版只让 LOCAL_RANK=0 写，多卡下会丢掉另外 3 张卡的 microbatch 形状，
  分子就没法按"全 world 的 token 数"累加。

计时边界严格按《MFU 实验统计原则》§4.1：
  1. 请求该 step 第一个 microbatch 之前开始计时（含数据等待）
     → patch Trainer.get_batch_samples
  2. 含全部 gradient-accumulation microbatch、fwd/bwd、optimizer、scheduler
  3. optimizer step 完成后 torch.cuda.synchronize() 再打结束时间戳
  4. 在 eval / 重日志 / checkpoint 之前结束 → on_step_end 天然满足

env：
  MFU_TIMING_OUT   输出 jsonl 路径模板（必填才启用）；实际写
                   <stem>_rank{LOCAL_RANK}<suffix>
  MFU_E2E_T_START  外层 launcher 记录的 T_start（epoch 秒）
"""

import os
import sys
import warnings

_VENV_SITE = "/fine-tuning-launcher/.venv/lib/python3.11/site-packages/sitecustomize.py"


def _exec_source(path, tag):
    if not os.path.isfile(path):
        warnings.warn(f"[mfu-sitecustomize] {tag} 源文件不存在: {path}")
        return
    try:
        g = {"__file__": path, "__name__": "__merged_%s__" % tag}
        with open(path, "r", encoding="utf-8") as f:
            code = compile(f.read(), path, "exec")
        exec(code, g)
    except Exception as _e:
        warnings.warn(f"[mfu-sitecustomize] exec {tag} 失败: {_e!r}")


# 1) venv 钩子：CCE / LoRAFusion / pin_shm（各自 env 门控；本实验两者都不设 → no-op）
_exec_source(_VENV_SITE, "venv")


# 2) per-step MFU 计时钩子
_TIMING_OUT = os.environ.get("MFU_TIMING_OUT", "").strip()
_LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def _rank_path(path, rank):
    root, ext = os.path.splitext(path)
    return f"{root}_rank{rank}{ext or '.jsonl'}"


def _log(msg):
    if _LOCAL_RANK == 0:
        print(f"[mfu-timing] {msg}", flush=True)


def _install_timing():
    import json
    import time

    out_path = _rank_path(_TIMING_OUT, _LOCAL_RANK)
    state = {
        "t_start": None,
        "microbatches": None,
        "fh": None,
    }

    def _open():
        if state["fh"] is None:
            d = os.path.dirname(out_path)
            if d:
                os.makedirs(d, exist_ok=True)
            state["fh"] = open(out_path, "w", buffering=1)
        return state["fh"]

    def _describe(inputs):
        """从一个 microbatch 的 inputs 抽 shape 与逐行真实长度。

        此处触发一次 device→host 拷贝；调用点在 step 起始、GPU 队列已被上一步的
        synchronize 排空，故开销 ~0.1ms 量级，且所有对照组协议一致。

        ★ 三种口径，必须分开处理，否则 attention core 的 ΣS² 会算错：
          1) packing=false      → attention_mask 是 2 维 0/1，逐行求和即真实长度
          2) neat_packing + FA3 → collator 把 attention_mask 置成 None（交给
             transformers 自己按 position_ids 生成 packed causal mask）。此时
             一行里装了多条样本，"真实长度"是**每个子段各自的长度**，不是整行。
             必须从 position_ids 反解：它每遇到新样本就归零，所以两次归零之间
             就是一段。若整行当成一条 4096，ΣS² 会虚高到几十倍。
          3) 都拿不到         → 标记 mask_present=False 让分析端报警，不静默糊过去
        """
        try:
            ids = inputs.get("input_ids")
            if ids is None:
                return None
            shape = list(ids.shape)
            mask = inputs.get("attention_mask")
            pos = inputs.get("position_ids")
            mode = None
            if mask is not None and mask.dim() == 2:
                lengths = [int(x) for x in mask.sum(dim=-1).tolist()]
                mode = "attention_mask"
            elif pos is not None and pos.dim() == 2:
                # neat_packing：position_ids 形如 [0,1,2,...,n1-1, 0,1,...,n2-1, 0,0,...]
                # 每次回到 0 就是一条新样本的开始；尾部 padding 也被写成一串 0。
                # 切段规则：p == 0 处开一段新段；padding 那串 0 会各自成为长度 1 的
                # 伪段，靠"段长 >= 2 才算真样本"滤掉（真实样本最短 424 token）。
                lengths = []
                for row in pos.tolist():
                    seg = 0
                    for p in row:
                        if p == 0:
                            if seg >= 2:
                                lengths.append(seg)
                            seg = 1
                        else:
                            seg += 1
                    if seg >= 2:
                        lengths.append(seg)
                mode = "position_ids"
            else:
                lengths = [int(shape[-1])] * int(shape[0])
                mode = "fallback_full_length"
            labels = inputs.get("labels")
            label_tokens = int((labels != -100).sum().item()) if labels is not None else None
            return {
                "shape": shape,
                "lengths": lengths,
                "length_source": mode,
                "mask_present": mode != "fallback_full_length",
                "padded_tokens": int(shape[0]) * int(shape[1]),
                "nonpad_tokens": int(sum(lengths)),
                "label_tokens": label_tokens,
            }
        except Exception as e:  # 记录失败不能拖垮训练
            return {"error": repr(e)}

    import torch
    import transformers.trainer as _T

    _orig_get_batch = _T.Trainer.get_batch_samples

    def _patched_get_batch_samples(self, *a, **kw):
        # ★ 计时起点：请求第一个 microbatch 之前
        state["t_start"] = time.monotonic()
        out = _orig_get_batch(self, *a, **kw)
        try:
            batch_samples = out[0] if isinstance(out, tuple) else out
            state["microbatches"] = [_describe(x) for x in batch_samples]
        except Exception:
            state["microbatches"] = None
        return out

    _T.Trainer.get_batch_samples = _patched_get_batch_samples

    import transformers.trainer_callback as _TC

    _Base = _TC.TrainerCallback

    class _MFUTimingCallback(_Base):
        def on_step_end(self, args, state_, control, **kw):
            # ★ 计时终点：optimizer/scheduler 已完成 → synchronize → 打时间戳
            #   本回调在 _maybe_log_save_evaluate 之前触发，eval/日志/ckpt 不入分母
            torch.cuda.synchronize()
            t_end = time.monotonic()
            t0 = state["t_start"]
            if t0 is None:
                return control
            step = int(getattr(state_, "global_step", 0))
            mbs = state["microbatches"] or []
            valid = [m for m in mbs if m and "error" not in m]
            rec = {
                "step": step,
                "local_rank": _LOCAL_RANK,
                "optimizer_step_time_s": t_end - t0,
                # 显存峰值：判断 OOM 余量用，不进 MFU 任何一项
                "peak_mem_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_mem_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "microbatch_count": len(mbs),
                "padded_tokens": sum(m["padded_tokens"] for m in valid),
                "nonpad_tokens": sum(m["nonpad_tokens"] for m in valid),
                "label_tokens": sum(
                    m["label_tokens"] for m in valid if m.get("label_tokens") is not None
                ),
                "microbatches": mbs,
            }
            fh = _open()
            if fh is not None:
                fh.write(json.dumps(rec) + "\n")
            state["t_start"] = None
            state["microbatches"] = None
            return control

    _orig_handler_init = _TC.CallbackHandler.__init__

    def _patched_handler_init(self, callbacks, *a, **kw):
        _orig_handler_init(self, callbacks, *a, **kw)
        if not any(isinstance(c, _MFUTimingCallback) for c in self.callbacks):
            self.add_callback(_MFUTimingCallback())

    _TC.CallbackHandler.__init__ = _patched_handler_init
    _log(f"per-step 计时钩子已安装（每 rank 各一份）→ {_rank_path(_TIMING_OUT, '*')}")


if _TIMING_OUT:
    # 惰性：等 llamafactory.train.tuner 完全 import 后再装，避免循环 import
    _state = {"done": False}

    def _b_get():
        return __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__

    _b = _b_get()
    _orig_import = _b["__import__"]

    def _hooked_import(name, *args, **kwargs):
        mod = _orig_import(name, *args, **kwargs)
        if not _state["done"]:
            tuner = sys.modules.get("llamafactory.train.tuner")
            if tuner is not None and hasattr(tuner, "run_exp"):
                _state["done"] = True
                _b["__import__"] = _orig_import
                try:
                    _install_timing()
                except Exception as e:
                    warnings.warn(f"[mfu-sitecustomize] 计时钩子安装失败: {e!r}")
        return mod

    _b["__import__"] = _hooked_import
