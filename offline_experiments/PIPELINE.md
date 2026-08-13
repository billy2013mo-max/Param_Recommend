# Staged experiment controller

> **历史控制器警告（2026-07-22）**：本文记录已经执行过的分阶段实验控制器，
> 不是当前待运行清单。`runtime/pipeline/pending-*.jsonl` 和旧 candidate/queue 是
> 可追溯记录，不表示仍需恢复执行；不要依据本文的 `--execute` 示例启动历史任务。
> 当前 H800 下一步是 export → historical recovery → readiness audit 后的离线理论
> 拟合与历史折叠验证。只有该验证暴露明确证据缺口时，才为最小补实验创建新 schema、
> 新审批和新执行说明。

`scripts/run_pipeline.py` is the resumable controller for the stages after the
memory-boundary experiment. Its default mode is read-only and never launches a
training process:

```bash
/fine-tuning-launcher/.venv/bin/python scripts/run_pipeline.py
```

The historical controller also had an execution mode. Its copy-pasteable launch
command is intentionally omitted here: old approvals, pending files, and the
steps below do not authorize a current resume. A future gap-specific campaign
must publish its own new-schema execution instructions after explicit approval.

The controller performs these gates in order:

1. Wait for or resume missing memory-boundary families.
2. Materialize every feasible throughput candidate and run a 10-warmup,
   20-measurement-step screen. A healthy existing formal result substitutes
   for the screen of the same physical configuration.
3. Preserve the fastest candidate at each measured GPU count, fill the
   remaining slots by throughput up to Top-3, then run only those finalists
   with the 20-warmup, 100-measurement-step formal window. Rerun only failures
   or successes with an incomplete measurement window/rank summary.
4. At each of 1, 2 and 4 GPUs, choose the fastest complete ZeRO/GC/MBS
   configuration by measured `samples/s` (median only for an explicitly
   repeated historical design). If a doubling gains less than 70%, do not
   schedule the next card count for that family.
5. Run packing MBS=1 memory probes and strict paired packing/no-packing
   measurements once per side. Enable packing only when the pair is healthy and
   its time gain reaches 10% (baseline MBS=1) or 20% (baseline MBS>1).
6. Run the four 8B calibration profiler jobs, derive independent holdout jobs
   from completed throughput winners, fit the FLOP correction, and evaluate
   holdout errors.
7. Write all decisions and validation metrics to
   `artifacts/stage_decisions.json` and the FLOP calibration to
   `artifacts/profiler_calibration.json`.

To materialize the current throughput-screen candidates into the resumable
queue without launching a scheduler or requiring approval:

```bash
/fine-tuning-launcher/.venv/bin/python scripts/run_pipeline.py --prepare-throughput
```

This writes `runtime/pipeline/pending-throughput-screen.jsonl` and records
`queued_waiting_approval` in the pipeline state.

The state file is `runtime/pipeline/state.json`. A controller lock prevents two
pipelines from running concurrently. Existing successful job IDs are removed
from resume inputs; explicit OOM is accepted only for memory probes. Ordinary
failures stop the phase gate and cannot silently become an infeasibility label.

The independent analysis command is:

```bash
/fine-tuning-launcher/.venv/bin/python scripts/stage_decisions.py
```

It is safe before results exist: unfinished decisions are reported as pending.
The resource-model validation uses leave-one
`(model, train_type, dataset, target_gbs)` scenario out, so repetitions or
alternative configs from the held-out scenario cannot leak into the fit.

## Approval transition after the running Stage A

Stage A was frozen before these two controller files existed. They were added
as new files so the active run's manifest remains valid. Do not execute the
downstream controller under that old manifest. Once Stage A completes, rerun
provenance capture and setup validation, review the new design SHA, and bind the
approval file to it. The controller enforces this and refuses `--execute` while
its own sources are absent from the frozen manifest.
