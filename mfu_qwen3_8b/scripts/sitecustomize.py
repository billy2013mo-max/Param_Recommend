# -*- coding: utf-8 -*-
"""合并委托 sitecustomize：venv 钩子(CCE/LoRAFusion/pin_shm) + per-step MFU 计时钩子。

计时边界严格按《MFU 实验统计原则》§4.1：
  1. 在请求该 step 的第一个训练 microbatch 之前开始计时
     → patch Trainer.get_batch_samples，在取数前打时间戳（因此包含数据等待）
  2. 包含全部 gradient-accumulation microbatch、forward/backward、optimizer、scheduler
  3. 在 optimizer step 完成后执行 CUDA synchronize
     → on_step_end 里 torch.cuda.synchronize() 后打结束时间戳
  4. 在 eval、日志重任务和 checkpoint callback 开始之前结束计时
     → on_step_end 在 _maybe_log_save_evaluate 之前被调用，天然满足

同时记录每个 microbatch 的真实 shape 与逐行真实长度，供 attention core 的 12*L*h*ΣS² 使用。

env：
  MFU_TIMING_OUT   输出 jsonl 路径（必填才启用）
  MFU_E2E_T_START  外层 launcher 记录的 T_start（epoch 秒），用于 E2E
激活后仅 LOCAL_RANK=0 写文件。
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


# 1) venv 钩子：CCE / LoRAFusion / pin_shm（各自 env 门控，未设即 no-op）
_exec_source(_VENV_SITE, "venv")


# 2) per-step MFU 计时钩子
_TIMING_OUT = os.environ.get("MFU_TIMING_OUT", "").strip()


def _log(msg):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(f"[mfu-timing] {msg}", flush=True)


def _install_timing():
    import json
    import time

    is_rank0 = int(os.getenv("LOCAL_RANK", "0")) == 0
    state = {
        "t_start": None,
        "microbatches": None,
        "records": [],
        "fh": None,
        "step_seen": 0,
    }

    def _open():
        if state["fh"] is None and is_rank0:
            os.makedirs(os.path.dirname(_TIMING_OUT), exist_ok=True)
            state["fh"] = open(_TIMING_OUT, "w", buffering=1)
        return state["fh"]

    def _describe(inputs):
        """从一个 microbatch 的 inputs 抽 shape 与 attention 计算所需的真实分段长度。

        此处会触发一次 device→host 拷贝；调用点在 step 起始、GPU 队列已被上一步
        的 synchronize 排空，故开销约 0.1ms 量级，且所有对照组协议一致。

        ★ 分段是 attention 分子的关键，不能只记每行总长 ★
        neat_packing 把多条样本拼进一行，但用块对角掩码让每条只能看到自己，所以
        attention 的 sum(S^2) 必须按**每条样本**算，不能按整行算。两者相差恰好是
        每包样本数（16k cutoff 下约 6 倍，短样本语料可达 40 倍以上）。
        三种 mask 形态各自的还原方式：
          1. neat_packing 预处理阶段把 attention_mask 写成分段 id 1,2,3,...（见
             `data/processor/supervised.py:214`），padding 为 0。此时 mask.sum()
             毫无意义（它在求 id 的和），必须按 id 分组统计每段长度。
          2. collator 对 FA2/FA3 路径会把 attention_mask 置 None（见
             `data/collator.py:532`），改由 position_ids 表达分段——每条样本的
             position_ids 从 0 重新开始，故 `position_ids == 0` 的位置就是段首。
          3. 普通 padding（packing=false）：mask 是 0/1，每行一条样本，sum 即真实长度。
        """
        try:
            ids = inputs.get("input_ids")
            if ids is None:
                return None
            shape = list(ids.shape)
            mask = inputs.get("attention_mask")
            position_ids = inputs.get("position_ids")
            lengths: list[int] = []
            segmentation = None

            if mask is not None and mask.dim() == 2:
                rows = mask.tolist()
                distinct = {int(v) for row in rows for v in row}
                if distinct - {0, 1}:
                    # 形态 1：分段 id。按非零 id 分组，每组长度即该样本长度。
                    segmentation = "neat_packing_segment_ids"
                    for row in rows:
                        counts: dict[int, int] = {}
                        for value in row:
                            value = int(value)
                            if value:
                                counts[value] = counts.get(value, 0) + 1
                        lengths.extend(counts[key] for key in sorted(counts))
                else:
                    # 形态 3：普通 0/1 padding mask。
                    segmentation = "binary_padding_mask"
                    lengths = [int(sum(int(v) for v in row)) for row in rows]
            elif position_ids is not None and position_ids.dim() == 2:
                # 形态 2：mask 已被置 None，用 position_ids 归零点切分。
                # 注意 padding 位的 position_ids 也被写成 0（见
                # `data/processor/supervised.py:221`），一串连续的 0 是 padding
                # 而不是一串长度为 1 的样本；只有"0 后面紧跟 1"才是真的新段起点。
                segmentation = "position_ids_resets"
                for row in position_ids.tolist():
                    values = [int(v) for v in row]
                    current = 0
                    for offset, value in enumerate(values):
                        following = values[offset + 1] if offset + 1 < len(values) else None
                        if value == 0 and offset > 0:
                            if following == 1:
                                # 真正的段首：结算上一段并开新段。
                                if current:
                                    lengths.append(current)
                                current = 1
                            else:
                                # padding 尾巴：结算上一段，剩下的全部丢弃。
                                if current:
                                    lengths.append(current)
                                current = 0
                        else:
                            current += 1
                    if current:
                        lengths.append(current)
            else:
                # 无分段信息可用：退化为整行长度。这会高估 packing 的 attention，
                # 故显式标注，让下游能判定该记录不可用于 attention 分子。
                segmentation = "unsegmented_row_fallback"
                lengths = [int(shape[-1])] * int(shape[0])

            labels = inputs.get("labels")
            label_tokens = int((labels != -100).sum().item()) if labels is not None else None
            return {
                "shape": shape,
                "lengths": lengths,
                "segmentation": segmentation,
                "segment_count": len(lengths),
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
            # ★ 计时终点：optimizer/scheduler 已完成 → CUDA synchronize → 打时间戳
            #   本回调在 _maybe_log_save_evaluate 之前被调用，故 eval/日志/ckpt 不入分母
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
                "optimizer_step_time_s": t_end - t0,
                # 显存峰值：判断 OOM 余量用，不进 MFU 任何一项
                "peak_mem_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_mem_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "microbatch_count": len(mbs),
                "padded_tokens": sum(m["padded_tokens"] for m in valid),
                "nonpad_tokens": sum(m["nonpad_tokens"] for m in valid),
                # 分段还原方式必须随记录留痕：unsegmented_row_fallback 意味着该步的
                # attention 分子会高估 packing，下游据此拒绝而不是静默算错。
                "segmentations": sorted(
                    {m["segmentation"] for m in valid if m.get("segmentation")}
                ),
                "segment_count": sum(
                    m["segment_count"] for m in valid if m.get("segment_count") is not None
                ),
                "label_tokens": sum(
                    m["label_tokens"] for m in valid if m.get("label_tokens") is not None
                ),
                "microbatches": mbs,
            }
            state["records"].append(rec)
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
    _log(f"per-step 计时钩子已安装 → {_TIMING_OUT}")


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
