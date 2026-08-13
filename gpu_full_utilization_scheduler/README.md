# GPU 满载调度策略

这个目录提供一个独立的静态调度器。它只生成调度计划，不启动训练、不卡 GPU，也不依赖项目里的实验调度器，因此可以安全地被后续实验复用。

## 解决什么问题

给定：

- 当前可用的 GPU ID；
- 本次最多允许使用的卡数；
- 每个任务需要的 GPU 数；

脚本会计算每个波次应该并发哪些任务以及每个任务的 GPU 掩码，并检查是否真的用满卡数上限。

同卡数任务的最大并发数是：

$$
并发数 = \left\lfloor \frac{卡数上限}{单任务卡数} \right\rfloor
$$

只有当“卡数上限能被单任务卡数整除”时，同卡数波次才能天然满载。例如 8 卡上，1、2、4、8 卡任务的并发数分别是 8、4、2、1；6 卡上单独运行 4 卡任务一定会空 2 卡。

## 两种策略

### 同卡数波次（homogeneous）

适合显存、吞吐和配置排序实验。一个波次只运行相同卡数的任务，避免把不同通信规模的任务混在一起。

- 优点：相同卡数下的测量条件更一致。
- 限制：如果卡数上限不能被任务卡数整除，数学上无法满载。
- 尾波：任务数量不能整除并发数时，可以报错、允许部分空闲，或者生成明确标记的重复测量补齐。

### 混合卡数装箱（mixed）

适合普通离线任务，允许一个波次混合不同卡数的任务。例如 6 卡可以运行一个 4 卡任务和一个 2 卡任务。

- 优点：更容易把卡用满。
- 限制：不同任务之间可能产生不同的通信和功耗干扰，不建议直接用于严格吞吐排序。

## 查看某个卡数上限的策略

8 张可用卡，任务规格是 1、2、4、8 卡：

```bash
python gpu_full_utilization_scheduler/full_gpu_scheduler.py strategy \
  --gpu-ids 0-7 \
  --gpu-limit 8 \
  --job-sizes 1,2,4,8
```

把上限改成 6 卡：

```bash
python gpu_full_utilization_scheduler/full_gpu_scheduler.py strategy \
  --gpu-ids 0-7 \
  --gpu-limit 6 \
  --job-sizes 1,2,4
```

输出会明确显示：1 卡任务并发 6 个、2 卡任务并发 3 个；4 卡同类波次只能使用 4 张卡，但混合策略可以用 4+2 填满。

`--gpu-ids` 支持不连续卡号，例如 `0,2,4-7`。`--gpu-limit` 会从这份可用列表中取前 N 张卡。

## 为任务清单生成调度计划

输入是 JSONL，每行至少包含：

```json
{"job_id": "job-a", "gpu_count": 1}
{"job_id": "job-b", "gpu_count": 1}
{"job_id": "job-c", "gpu_count": 2}
```

生成严格同卡数计划，尾波不能满载就报错：

```bash
python gpu_full_utilization_scheduler/full_gpu_scheduler.py plan \
  --jobs jobs.jsonl \
  --gpu-ids 0-7 \
  --gpu-limit 8 \
  --policy homogeneous \
  --tail-policy error \
  --output schedule.json
```

允许生成重复测量补齐尾波：

```bash
python gpu_full_utilization_scheduler/full_gpu_scheduler.py plan \
  --jobs jobs.jsonl \
  --gpu-limit 8 \
  --policy homogeneous \
  --tail-policy repeat \
  --output schedule.json
```

脚本生成的补齐任务都会带上：

- `synthetic_repeat: true`
- `repeat_of_job_id`
- `ranking_eligible: false`

脚本不会自动执行这些重复任务。调用方必须把它们正式写进实验设计并重新冻结批准，不能在实验运行过程中偷偷追加。

生成混合装箱计划：

```bash
python gpu_full_utilization_scheduler/full_gpu_scheduler.py plan \
  --jobs jobs.jsonl \
  --gpu-limit 6 \
  --policy mixed \
  --tail-policy error \
  --output schedule.json
```

如果只想尽量利用、不要求每个波次满载，需要同时声明：

```bash
--tail-policy partial --allow-partial
```

## 失败条件

脚本在以下情况下直接报错：

- 任务卡数超过卡数上限；
- GPU ID 重复或卡数上限超过可用卡数；
- 严格同卡数策略在当前上限下数学上无法满载；
- 尾波不满且没有明确选择 `repeat` 或 `partial`；
- 混合策略也找不到能够填满卡数上限的任务组合。

例如卡数上限是 5，而任务只有 2 卡和 4 卡规格，则任何组合都是偶数，不可能用满 5 卡。脚本会报错，不会输出一个伪满载计划。

## 测试

```bash
python -m unittest -v gpu_full_utilization_scheduler.test_full_gpu_scheduler
```
