#!/usr/bin/env python3
"""修复 Qwen3.5 sdpa 矩阵里 4 条 tok/s 解析错误的记录。

背景
----
`train_tokens_per_second[^0-9]*([0-9.]+)` 这个正则在**真值 ≥ 10⁴ 时一律把结果
除以 10⁴**。原因是 HuggingFace 把大浮点数按科学计数法打印（`1.193e+04`），
而字符类 `[0-9.]+` 在 `e` 处停下，捕获到 `1.193` 就丢掉了 `e+04`。

所以判据很明确：**真值 < 10⁴ 的行不受影响，≥ 10⁴ 的行全错**。这也解释了
为什么本文件 144 行里只有 4 条坏（其余行的真值都小于一万），以及为什么坏值
恰好都落在 1.19～1.72 这个区间（对应真值 11915～17206）。

VL 侧 24 条已在 2026-08-25 修过，但 Qwen3.5 sdpa 矩阵里这 4 条当时没覆盖
（`source` 仍是 `log_regex_only`）。

顺带一个更重要的发现：在**当前**环境下 `train_tokens_per_second` 这个字段
已经完全不存在于 metrics 里（`all_results.json` 里是 None），正则会匹配到
日志中别处的数字。新跑的作业不能再依赖这个正则，必须用
`num_input_tokens_seen / train_runtime` 直接算——见
`recompute_tps_from_results.py`。

修复方式
--------
本文件对应的作业输出目录已不在，无法读 `all_results.json`，因此用
`samples_per_s` 与 `tokens_seen` 反算：

    tok/s = samples_per_s × tokens_seen / (总步数 × 卡数 × mbs × ga)

其中总步数 = warmup + measure。这个公式在 225 条未受影响的 log_regex_only
行上逐一验证过，偏差全部 < 5%（多数 < 0.1%），可以放心用来重建这 4 条。
重建值 11915～17206 与「坏值 × 10⁴」完全吻合，互为交叉验证。

只改这 4 条，其余行原样保留；改动的行加 `source="recomputed_from_samples_per_s"`
和 `tokens_per_s_original` 留痕。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "artifacts" / "rtx4090_qwen35_observations_20260825"
TARGET = DATA_DIR / "qwen35_sdpa_matrix.jsonl"

REL_TOL = 0.05   # 与重算值偏差超过这个比例才认为是坏行


def recompute(r: dict) -> float | None:
    """用 samples_per_s 与 tokens_seen 反算 tok/s。缺字段返回 None。"""
    sps, seen = r.get("samples_per_s"), r.get("tokens_seen")
    if not sps or not seen:
        return None
    steps = r.get("warmup", 0) + r.get("measure", 0)
    denom = steps * r["gpu_count"] * r["mbs"] * (r.get("ga") or 1)
    if denom <= 0:
        return None
    return sps * seen / denom


def main():
    rows = [json.loads(l) for l in TARGET.open()]
    fixed, checked = [], 0
    out = []
    for r in rows:
        if (r.get("source") == "log_regex_only" and r["exit"] == 0
                and not r["oom"] and r.get("tokens_per_s")):
            rc = recompute(r)
            if rc is not None:
                checked += 1
                rec = r["tokens_per_s"]
                if abs(rc - rec) / max(rec, 1e-9) > REL_TOL:
                    r = dict(r)
                    r["tokens_per_s_original"] = rec
                    r["tokens_per_s"] = round(rc, 3)
                    r["source"] = "recomputed_from_samples_per_s"
                    fixed.append((r["tag"], rec, rc))
        out.append(r)

    print(f"检查了 {checked} 条 log_regex_only 记录，发现 {len(fixed)} 条需要修正：")
    for tag, old, new in fixed:
        print(f"  {old:9.2f} → {new:9.0f} tok/s  ({new/old:.0f}x)  {tag}")

    if not fixed:
        print("无需修改。")
        return

    backup = TARGET.with_suffix(".jsonl.bak_before_tps_repair")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"\n原文件已备份到 {backup.name}")
    with TARGET.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已写回 {TARGET.name}（{len(out)} 行）")


if __name__ == "__main__":
    main()
