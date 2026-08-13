# Qwen3-14B Full SFT 两卡配置对齐实测

实验日期：2026-07-30  
硬件：2 × NVIDIA H800（每组固定使用同一节点内的 GPU）  
任务：Qwen3-14B Full SFT，`cutoff_len=2048`，普通 Packing，45 个优化器步  
全局 Batch：三组均为 128

## 结论

找到了一个安全且略快的候选，但提升很小：

- 原配置：部分 optimizer offload，`MBS=8 / GA=8`。
- 推荐候选：保持完全相同的部分 optimizer offload，仅改为 `MBS=16 / GA=4`。
- 推荐候选的 45 步训练累计吞吐提高 **1.06%**；去掉首步热身后、保留 epoch 边界的稳定段提高 **2.19%**。
- 推荐候选的峰值显存约 **123.48 GiB/卡**，相对 132.84 GiB 的当前安全线仍有 **9.36 GiB/卡** 余量。
- 原配置和推荐配置的最终训练 loss 几乎相同：`1.943141` 与 `1.943129`。

因此，若目标是“在安全前提下尽量提高训练吞吐”，可以使用 `MBS=16 / GA=4`；但它不是显著优化。若产品要求候选至少提高 3%–5% 才切换，则本次结果不足以替换原配置。

不开 offload 的 `MBS=2 / GA=32` 虽然 45 步累计吞吐比原配置高 2.78%，但峰值显存达到 **136.07 GiB/卡**，超过 132.84 GiB 安全线 3.24 GiB，并出现一次 allocator cache flush。这个收益不足以抵消 OOM 风险，不建议作为默认配置。

## 严格对齐项

三组实验保持以下项目不变：

- 模型、数据集、模板、`cutoff_len`、Packing；
- 2 卡、全局 Batch 128、45 步；
- FA3、Liger、CCE、Gradient Checkpointing；
- 学习率、调度器、优化器名称、随机种子；
- 训练和保存策略。

部分 offload 两组使用用户给出的同一个 DeepSpeed 文件：

`/wanqing-develop/chengjin/z3-offload实验/configs/ds_z3_antibubble.json`

其 SHA-256 为：

`5683dabed33600858b609a89e34f9a0c9825ac4d034ab59953de12d4a1ea2ace`

其中 `offload_optimizer.ratio=0.4` 表示 ZeRO-3 的部分 optimizer offload：约 40% 走 CPU Adam；配置没有 `offload_param`，因此并未把模型参数本身 offload 到 CPU。运行时统一设置 `OMP_NUM_THREADS=48`。

## 实测指标

| 配置 | Offload | MBS / GA | 45 步累计 token/s | 稳定段 token/s | 完整 Batch 稳定 token/s | 完整步中位数 | 峰值显存/卡 | 安全线余量 | Cache flush | 最终 train loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 YAML | optimizer 40% CPU | 8 / 8 | 8,251.24 | 8,272.79 | 8,349.27 | 30.784 s | 121.38 GiB | +11.45 GiB | 0 | 1.943141 |
| 推荐候选 | optimizer 40% CPU | 16 / 4 | 8,338.35 | 8,453.84 | 8,529.98 | 29.905 s | 123.48 GiB | +9.36 GiB | 0 | 1.943129 |
| no-offload 对照 | 无 | 2 / 32 | 8,480.75 | 8,501.84 | 8,503.76 | 30.793 s | 136.07 GiB | -3.24 GiB | 1 | 1.659888 |

指标口径：

- **45 步累计 token/s**：最后一个优化器步日志中的累计输入 token / 训练主体时间，包含首步热身与 epoch 边界，不包含 checkpoint 保存。
- **稳定段 token/s**：从第 6 步开始，按实际 token 增量除以墙钟增量；保留 epoch 尾部小 Batch 和下一 epoch 的启动开销。
- **完整 Batch 稳定 token/s**：稳定段中只保留 token 增量为 262,144 的完整优化器步，但仍保留跨 epoch 后第一个完整步的开销。
- **安全线余量**：`132.839 GiB - 实测峰值显存`；负数表示虽未 OOM，但不满足当前产品安全缓冲。

## 相对变化

推荐候选相对原 YAML：

- 45 步累计吞吐：**+1.06%**
- 包含 epoch 边界的稳定段吞吐：**+2.19%**
- 完整 Batch 稳定吞吐：**+2.16%**
- 峰值显存：**+2.09 GiB/卡**

no-offload 对照相对原 YAML：

- 45 步累计吞吐：**+2.78%**
- 包含 epoch 边界的稳定段吞吐：**+2.77%**
- 完整 Batch 稳定吞吐：**+1.85%**
- 峰值显存：**+14.69 GiB/卡**

## 为什么普通步更快，但 45 步只提高 1%

`MBS=16 / GA=4` 的普通完整步通常约为 29.7–30.1 秒，原配置通常约为 30.7–30.9 秒；但本数据集 Packing 后只有 1,720 个训练样本，每约 14 个优化器步就跨一次 epoch。

跨 epoch 时会出现：

1. 一个 token 数较少的尾部 Batch；
2. 下一 epoch 第一个完整步的额外启动开销。

推荐配置在连续普通步上的收益，被首步热身和三个 epoch 边界长尾吃掉了一部分。因此短至 45 步的累计收益只有 1.06%；对 epoch 更长、训练步更多的任务，稳定段约 2.2% 的结果更有代表性。

## no-offload 不能简单视为等价替换

no-offload 对照把 MBS 降到 2，并把 GA 提到 32，保持全局 Batch 128。它证明了不开 offload 时可以通过减小 MBS 跑通，但有三点限制：

1. 峰值显存约 136.07 GiB/卡，物理上只剩约 4.33 GiB，且超过产品安全线；
2. 出现过一次显存压力 cache flush；
3. 最终 train loss 与两条部分 offload 路线明显不同，说明优化器数值轨迹发生了变化。更低的训练 loss 不能直接解释成更高的验证质量。

因此它适合作为诊断对照，不适合作为当前默认推荐。

## Predictor 暴露出的缺口

当前正式 `physical-shares + v4b` predictor 没有建模：

- `packing=true`；
- ZeRO-3 的部分 optimizer offload；
- `offload_optimizer.ratio` 和 CPU Adam 线程数。

所以 predictor 把本次 10 个影子候选全部过滤，包括已经实测跑通的原 YAML。此次 `MBS=16 / GA=4` 候选来自相同运行路径的历史实测，而不是 predictor 自动放行。

这三次新结果可作为后续版本化历史锚点的候选证据，但在完成复测和发布审查前，不应自动覆盖显存模型。

## 可比性限制

- 按用户追加要求，原配置在 GPU 4/5 上运行时，no-offload 对照同时在 GPU 6/7 上运行；两者共享节点 CPU、内存带宽和文件系统。
- 推荐候选随后单独运行。因此 1%–2% 的小差异还不能视为严格统计显著。
- 如果要做“是否替换默认配置”的严格决策，应将原配置与推荐候选在相同空闲条件下顺序重复至少 2 次，再比较配对中位数。
- 原配置的训练与最终保存均成功；其 wrapper 在最终保存后因运行期间扩展脚本而未写入 `exit_code` 元数据，但日志完整包含 45 步、`Training completed` 与 `final_save_model`。

## 文件

- `baseline_mbs8_ga8.yaml`：原 YAML 对齐配置
- `recommended_mbs16_ga4.yaml`：推荐候选
- `no_offload_mbs2_ga32.yaml`：no-offload 对照
- `ds_z3_no_offload.json`：仅移除 optimizer offload 的 ZeRO-3 配置
- `results.json`：逐步指标、稳定段统计和显存采样汇总
- `logs/`：完整训练日志和 2 秒间隔 GPU 采样
- `predictor_output.json`：正式 predictor 的影子回放结果

完整模型与 DeepSpeed checkpoint 输出原占用约 540 GiB。按用户要求，`outputs/` 已于 2026-07-30 永久删除；配置、日志、GPU 采样、报告与 `results.json` 均保留。
