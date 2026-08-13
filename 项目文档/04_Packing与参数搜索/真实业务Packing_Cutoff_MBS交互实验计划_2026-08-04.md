# 真实业务数据 Packing、Cutoff 与 MBS 交互实验计划

> 日期：2026-08-04  
> 状态：`materializing_for_execution`，按用户 2026-08-04 授权物化 40 项正式矩阵  
> 目标：回答“Neat Packing 固定 MBS=1 并扩大 cutoff”与“关闭 Packing 并扩大 MBS”在不同真实数据分布下如何取舍。

## 1. 要回答的问题

本实验不再只问固定 cutoff 下 Packing 是否更快，而是同时估计四个效应：

1. 非 Packing 下，把 MBS 从 1 放大到 `k` 的收益；
2. 相同 cutoff、MBS=1 下，开启 Neat Packing 的纯机制收益；
3. MBS=1 时，把 cutoff 从 `C` 放大到 `kC` 的成本和 Packing 收益变化；
4. 在名义动态 token 槽位相同的条件下，直接比较：

```text
非 Packing：MBS=k，cutoff=C
Packing：    MBS=1，cutoff=kC
```

两者都有 `MBS × cutoff = kC`，但训练计算、Padding、Attention、GA 和 token
保留率不同。本实验要测量这些差异，而不是假设两者等价。

## 2. 实验边界

主矩阵只使用纯文本 SFT 数据。以下数据不进入本轮：

- 7,108 条和 2,011 条双图数据：会引入视觉 encoder、图片 token 和 VL collator；
- 54,142 条偏好对数据：属于 DPO/ORPO，不是普通 SFT。

主矩阵统一为：

| 项目 | 冻结值 |
|---|---|
| 模型 | Qwen3-8B |
| 训练类型 | LoRA-SFT，rank/alpha=32，target=all |
| 硬件 | 1×H800 140GB |
| 数据类型 | 纯文本真实业务数据 |
| dtype/kernel | BF16、FA3、Liger，绑定正式运行时快照 |
| DeepSpeed/offload | 均关闭 |
| GC | 主矩阵固定开启，保证长 cutoff 配置有共同可行域 |
| 目标 GBS | 组内冻结为 56 / 56 / 60 / 60 个逻辑样本，见 4.1 |
| Neat Packing | 开启时物理 MBS 固定为 1 |
| 预处理 worker | 8，与当前 LLaMA-Factory 装箱行为一致 |
| 测量 | warmup 2 + measure 8 optimizer steps |
| 重复 | 每个正式配置 2 次；满足扩展条件时补第 3 次 |

固定 GC 的目的是隔离 `Packing/cutoff/MBS`，不是声明 GC 开启一定是产品最快配置。
主矩阵完成后，产品确认阶段才允许各分支分别选择 GC on/off。

## 3. 真实业务数据分组

最终物化前，四份原始数据都必须使用同一个 Qwen3-8B tokenizer 和
`qwen3_nothink` template 重新画像。下表统计来自当前已存在画像；前两份当前绑定
Qwen3.5 tokenizer，因此 cutoff 只是设计候选，重新画像后如不满足语义门槛必须调整。

| 实验标识 | 条数 | 当前长度特征 | 基础 cutoff `C` | 宽 cutoff `kC` | `k` | 当前基础/宽 token 保留率 |
|---|---:|---|---:|---:|---:|---:|
| `177870条-极短集中` | 177,870 | mean 209，P50 199，P99 355，max 4,299 | 1,024 | 4,096 | 4 | 99.08% / 约 100% |
| `71014条-短文本长尾` | 71,014 | mean 565，P50 536，P99 968，max 13,709 | 4,096 | 16,384 | 4 | 99.54% / 100% |
| `4500条-教育会话中长` | 4,500 | mean 2,638，P50 2,525，P99 4,738，max 10,972 | 8,192 | 32,768 | 4 | 99.97% / 100% |
| `4500条-内容评测宽长尾` | 4,500 | mean 1,041，P50 661，P99 6,282，max 28,799 | 16,384 | 32,768 | 2 | 99.74% / 100% |

数据来源必须绑定现有下载清单中的原始文件 SHA256。报告与图表使用“条数＋分布标签”
作为数据标识，不使用 Blobstore 内部 ID 作为用户可见名称。

### 3.1 固定训练切片

- 177,870 条和 71,014 条：固定 seed，从完整数据中均匀无放回抽取 8,192 条；
- 两份 4,500 条数据：使用完整数据；
- 切片在看到 GPU 结果前冻结 JSONL、顺序和 SHA256；
- 不按长度重采样，不人为提高长尾占比；
- 本轮 GPU 可比性画像与训练都使用同一冻结切片；完整数据画像仅作为外部有效性描述。

## 4. 每组的五个主实验臂

对每份数据，按其 `C` 和 `k` 运行以下五个配置：

| 臂 | Packing | cutoff | 物理 MBS | GA 规则 | 用途 |
|---|---|---:|---:|---|---|
| `N-C-1` | 关 | `C` | 1 | `G_dataset / DP` | 非 Packing 基线 |
| `N-C-k` | 关 | `C` | `k` | `G_dataset / (DP×k)` | 单独测 MBS 放大收益 |
| `P-C-1` | 开 | `C` | 1 | 按平均逻辑样本/pack 派生 | 相同 cutoff 的 Packing 效应 |
| `N-kC-1` | 关 | `kC` | 1 | `G_dataset / DP` | 单独测 cutoff 放大成本 |
| `P-kC-1` | 开 | `kC` | 1 | 按平均逻辑样本/pack 派生 | 扩大 cutoff 后的 Packing 路线 |

关键直接比较为：

```text
N-C-k  vs  P-kC-1
```

这两个臂具有相同的名义 `MBS × cutoff`。同时保留另外三个臂，才能区分最终差异来自
MBS、cutoff，还是 Packing 本身。

四份数据 × 五个实验臂 × 两次重复，共 40 个正式任务。

### 4.1 组内逻辑 GBS 修正

Packing 的每个物理样本包含可变数量的逻辑样本，而 GA 必须是整数。统一强制 GBS=64
会使三个宽 cutoff Packing 臂产生约 6%–10% 的逻辑 GBS 偏差，违反第 5 节的 5% 门槛。
因此在看到 GPU 结果之前，把每组所有五个臂共同冻结到最接近且两条路线均可实现的目标：

| 数据分布 | `G_dataset` | 两个 Packing cutoff 的期望逻辑 GBS 最大偏差 |
|---|---:|---:|
| 177870条-极短集中 | 56 | <= 5% |
| 71014条-短文本长尾 | 56 | <= 5% |
| 4500条-教育会话中长 | 60 | <= 5% |
| 4500条-内容评测宽长尾 | 60 | <= 5% |

该修正只改变 GA，不改变五个 arm、cutoff、MBS、Packing 或 40 项任务总数。

### 4.2 宽长尾额外压力对（本轮暂缓）

`4500条-内容评测宽长尾` 额外保留一个不满足默认 99% token 保留门槛的诊断对：

| 臂 | Packing | cutoff | MBS | 当前 token 保留率 |
|---|---|---:|---:|---:|
| `N-8192-4-diagnostic` | 关 | 8,192 | 4 | 98.03% |
| `P-32768-1-diagnostic` | 开 | 32,768 | 1 | 100% |

两臂名义 token 槽位相同，正好对应四倍 MBS 与四倍 cutoff。该诊断对不满足 99% token
保留门槛，当前用户授权只要求 40 项主矩阵，因此这 4 项不在本轮队列中。

## 5. GPU 前静态物化

每个 `(数据, cutoff)` 在开跑前必须生成：

- 截断样本数与截断率；
- token 保留率；
- 截断后的 P50/P90/P95/P99/max；
- 8-worker 精确 greedy knapsack 的 pack 数量、利用率和每 pack 样本数分布；
- 非 Packing 在 MBS=1 和 MBS=k 时的 padding 利用率；
- effective/computed tokens 与 attention token pairs；
- Packing GA、期望逻辑 GBS 和相对误差。

正式主矩阵要求：

```text
样本截断率 <= 1%
token保留率 >= 99%
期望逻辑GBS相对误差 <= 5%
cutoff <= 模型最大上下文
```

诊断压力对明确豁免 token 保留率门槛，但必须在结果中标红，不参与产品胜负统计。

## 6. 显存与失败处理

本轮按用户要求只运行冻结的 40 项正式任务，不另加 canary 任务。CUDA allocator OOM
记为有效的右删失边界，不伪造峰值，也不临时更改 GC/MBS 破坏配对；非 CUDA OOM、
数据错误和启动错误不算显存边界，必须修复后按原 job SHA 重跑。

## 7. 指标与估计量

### 7.1 每个任务必须记录

- peak allocated、peak reserved、NVIDIA device used；
- logical samples/s；
- effective tokens/s、computed tokens/s；
- measured effective/computed tokens；
- attention token pairs/s；
- optimizer step time、micro-step time、GA；
- 实际逻辑 GBS 及相对目标误差；
- Packing 多样本 feature 数、边界/position/attention mask 违规数；
- consumed token ledger，排除 dataloader 预取未消费 batch；
- 由完整数据条数推算的 epoch wall time。

### 7.2 核心效应

按每份数据分别计算：

```text
MBS收益
  = throughput(N-C-k) / throughput(N-C-1) - 1

基础cutoff下Packing收益
  = throughput(P-C-1) / throughput(N-C-1) - 1

宽cutoff下Packing收益
  = throughput(P-kC-1) / throughput(N-kC-1) - 1

最终路线取舍
  = epoch_time(N-C-k) / epoch_time(P-kC-1) - 1
```

最终路线取舍必须和两边 token 保留率一起报告，不允许只给一个速度数字。

另外计算 cutoff 对 Packing 的差分效应：

```text
interaction =
  [log throughput(P-kC-1) - log throughput(N-kC-1)]
  - [log throughput(P-C-1) - log throughput(N-C-1)]
```

它回答“扩大 cutoff 后，Packing 相对非 Packing 是变得更有利还是更不利”。

## 8. 重复、顺序和停止规则

- 每个配置先执行 2 次，按数据组交错和反平衡顺序运行；
- 使用启动时空闲的 GPU 0、1、4、5、6、7，最多六个单卡任务并行；GPU 2、3 明确排除；
- 每个任务记录 GPU assignment；报告重复 CV，并把共享 CPU/I/O 争用列为小差异解释限制；
- 任一 treatment 的 CV >5%，补第 3 次；
- 两条路线性能差距绝对值 <5%，补第 3 次；
- 两次结果已显示某路线保守下界领先 >=20%，且两边健康，则不增加到第 4 次；
- 所有阈值、顺序、数据切片和任务 SHA 必须在首个 GPU job 前冻结。

## 9. 如何判定“哪条路线更好”

采用 Pareto 判定，不把质量与速度强行压成未校准的单一分数：

1. 先过滤 token 保留率、截断率、GBS 和显存不合格的 DataPlan；
2. 若 `P-kC-1` token 保留不少且 epoch 时间更短，它严格支配 `N-C-k`；
3. 若 Packing 保留更多 token 但训练更慢，输出质量/时间 Pareto，不自动替用户做取舍；
4. 只有两边处于同一语义门槛时，才按训练时间选胜者；
5. 自动开启 Packing 仍要求成对收益的保守下界达到发布阈值，不使用单次点估计。

## 10. 本轮能够与不能够得出的结论

本轮能够回答：

- 不同真实长度分布下，扩大 MBS 和扩大 Packing cutoff 谁更有利；
- 相同名义动态 token 槽位是否意味着相同显存；
- 哪些静态画像指标能够解释赢家；
- 当前静态 Packing 规则在哪些分布上产生假阴性或假阳性。

本轮不能直接证明：

- 结论可迁移到 VL、DPO、4090 或非 Qwen 模型；
- 四份数据足以训练高维通用模型；
- 绝对吞吐模型已经可以跨任意 cutoff 发布。

这 40 个任务首先用于估计交互面和修正规则。若要发布泛化模型，必须再用完全未参与
阈值选择的 Blobstore 新数据做 prospective holdout。

## 11. 后续迁移确认

主矩阵完成后，仅选择“极短集中”和“宽长尾”两个端点，在 Qwen3-14B Full、
2×H800、ZeRO-3、GC on 下复验最有信息量的三个臂：

```text
N-C-k
N-kC-1
P-kC-1
```

该阶段用于判断数据分布效应是否会被模型规模和 Full/LoRA 机制反转，不与主矩阵一起
拟合。是否执行以及 cutoff 是否缩小，取决于主矩阵显存 canary，不在当前设计中自动开跑。
