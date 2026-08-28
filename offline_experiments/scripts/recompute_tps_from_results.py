#!/usr/bin/env python3
"""用作业输出目录里的 all_results.json 重算 tok/s，覆盖日志正则的结果。

为什么必须这么做
----------------
矩阵脚本用 `re.findall(r"train_tokens_per_second[^0-9]*([0-9.]+)", log)` 取吞吐，
有两个问题：

1. **科学计数法截断**：HuggingFace 把大浮点数打印成 `1.193e+04`，而 `[0-9.]+`
   在 `e` 处停下，于是真值 ≥ 10⁴ 的行一律被除以 10⁴。
2. **字段已不存在**：在当前环境下 metrics 里根本没有 `train_tokens_per_second`
   （`all_results.json` 里是 None），正则会匹配到日志别处的数字，结果不可信。

权威算法只有一个：

    tok/s = num_input_tokens_seen / train_runtime

两个字段都在每个作业的 `all_results.json` 里（`trainer_state.json` 的
log_history 末条也有），是 Trainer 自己记的，不经过任何字符串解析。

用法
----
    python3 recompute_tps_from_results.py --run-dir /tmp/qwen35_oos_v1

会就地重写 `<run-dir>/results.jsonl`，把每行的 `tokens_per_s` 换成权威值，
原值留在 `tokens_per_s_log_regex`，并标 `tps_source="all_results"`。
原文件备份为 `results.jsonl.bak_before_tps_recompute`。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REL_TOL = 0.02   # 与正则值偏差超过这个比例才算「修正」，否则只是标记来源


def authoritative_tps(run_dir: Path, tag: str) -> tuple[float | None, str]:
    """从作业输出目录读权威 tok/s。返回 (值, 说明)。"""
    out = run_dir / f"out_{tag}"
    for fname in ("all_results.json", "trainer_state.json"):
        p = out / fname
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if fname == "trainer_state.json":
            hist = d.get("log_history") or []
            d = hist[-1] if hist else {}
        seen, rt = d.get("num_input_tokens_seen"), d.get("train_runtime")
        if seen and rt:
            return seen / rt, fname
    return None, "缺失"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results = args.run_dir / "results.jsonl"
    rows = [json.loads(l) for l in results.open()]

    fixed, marked, missing = [], 0, []
    out = []
    for r in rows:
        # OOM / 框架失败的作业没有完整 metrics，跳过
        if r["oom"] or r["exit"] != 0:
            out.append(r)
            continue
        true, src = authoritative_tps(args.run_dir, r["tag"])
        if true is None:
            missing.append(r.get("note", r["tag"]))
            out.append(r)
            continue
        old = r.get("tokens_per_s")
        r = dict(r)
        r["tokens_per_s_log_regex"] = old
        r["tokens_per_s"] = round(true, 3)
        r["tps_source"] = f"all_results:{src}"
        if old and abs(true - old) / max(old, 1e-9) > REL_TOL:
            fixed.append((r.get("note", r["tag"]), old, true))
        else:
            marked += 1
        out.append(r)

    print(f"共 {len(rows)} 行；正则值与权威值一致 {marked} 行，"
          f"修正 {len(fixed)} 行，缺 metrics {len(missing)} 行")
    for note, old, new in fixed:
        ratio = f"{new/old:.0f}x" if old else "—"
        print(f"  修正 {note:32} {old:12.3f} → {new:12.1f}  ({ratio})")
    for note in missing:
        print(f"  缺 metrics（保持原值）：{note}")

    if args.dry_run:
        print("\n--dry-run，未写回")
        return

    backup = results.with_suffix(".jsonl.bak_before_tps_recompute")
    if not backup.exists():
        shutil.copy2(results, backup)
        print(f"\n原文件已备份为 {backup.name}")
    with results.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已写回 {results}")


if __name__ == "__main__":
    main()
