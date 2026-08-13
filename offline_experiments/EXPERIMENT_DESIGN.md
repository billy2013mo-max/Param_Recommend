# H800 SFT 训练效率离线实验设计（待审批）

## 1. 本轮边界

- 任务：仅文本 SFT；训练类型为 FULL、LoRA；BF16。
- 当前开发机阶段模型：Qwen3-1.7B、4B、8B、14B。0.6B、32B、72B 不进入本轮正式矩阵。
- 硬件后续：0.6B/1.7B 补 RTX 4090 结果；1.7B 本轮仍保留作 H800 跨硬件标定；72B 所需的 8 卡或多机阶段暂不执行。
- 硬件：当前可用卡池为本机 H800 的 1–4 卡，卡数为 1、2、4；卡池可由冻结配置显式调整。
- 固定实现：当前 `/fine-tuning-launcher/.venv`、FA3、Liger、`adamw_torch_fused`，不把实现版本当搜索变量。
- 单卡不使用 DeepSpeed；多卡只使用 ZeRO-2/ZeRO-3；不使用 ZeRO-1 或 offload。
- `cutoff_len = ceil(max_tokens / 512) * 512`，packing on/off 使用同一值，不主动截断实验切片。
- LoRA rank/alpha 固定为 32，target=`all`，本轮不把 rank 当搜索维度。

训练仍由审批锁阻止，当前文件只定义实验，不会自动开跑。

## 2. 已修复并冻结的环境口径

1. GC 关闭配置同时写入：
   - `gradient_checkpointing: false`
   - `disable_gradient_checkpointing: true`

   这是必要修复；只写前者时，当前 LLaMA-Factory 仍可能启用 GC。

2. 单卡配置完全移除 `deepspeed`，不再用 ZeRO-0 模拟单卡。
3. FA3 固定 `FA3_VARIANT=orig`；CCE 固定关闭；Liger 固定开启，与生产 launcher 保持一致。
4. ZeRO-2/3 使用生产 launcher 的默认参数，`overlap_comm=false`。历史定制 overlap 配置不混入首轮标签。
5. 每个配置使用全新进程；OOM 后不复用进程；正式吞吐运行不保存模型权重。
6. no-packing 的 MBS 为 1 起按 2 倍递增；neat packing 的物理 MBS 固定为 1。
7. 正式任务启动时重新校验冻结设计中的全部文件 SHA256；冻结后任一脚本、配置、数据或模型清单变化都会使批准失效。
8. 批量实验前先通过四个 1.7B/512-token smoke case：单卡 LoRA、单卡 neat packing、双卡 FULL+ZeRO-2+GC、四卡 LoRA+ZeRO-3。
9. 复现身份同时绑定源码快照、模型目录名/路径、数据 SHA256、包版本、CUDA/NCCL、launcher commit 和容器运行身份。模型目录由平台约定保证正确，不读取模型文件计算 SHA256。平台若注入 image digest 则一并记录；无 Pod RBAC 时使用完整运行时指纹作为可验证替代。
10. 所有实验统一使用 Qwen3 tokenizer 与 `qwen3_nothink` 模板。Qwen2.5-72B 只提供 72B 权重；运行时目录用符号链接组合该权重目录与 Qwen3-8B tokenizer，不复制或散列权重。

## 3. 数据设计

每个切片 1000 条，统一使用当前 LLaMA-Factory 的 Qwen3 tokenizer 与 `qwen3_nothink` Template 编码和复核。

| 切片 | 类型 | Qwen3 P50 / P99 / Max | cutoff_len | 来源/处理 |
|---|---|---:|---:|---|
| `short_512` | 短且集中 | 130 / 431 / 448 | 512 | Alpaca Cleaned |
| `multiturn_2048` | 中等多轮 | 1094 / 1984 / 1984 | 2048 | UltraChat；超长内容做可追溯前缀裁剪 |
| `multiturn_4096` | 中等多轮 | 1094 / 3155 / 3916 | 4096 | UltraChat |
| `longtail_8192` | 长尾 | 244 / 8128 / 8128 | 8192 | 700 Alpaca + 250 UltraChat + 50 LongAlpaca |
| `longcontext_16384` | 长上下文边界 | 7959 / 16320 / 16320 | 16384 | LongAlpaca；边界受控裁剪 |
| `longcontext_32768` | 自然长上下文 | 7959 / 26673 / 32704 | 32768 | LongAlpaca；只裁剪超过 32K 档位的样本 |

源数据缓存和派生切片均记录 SHA256。新的在线下载路径使用 `datasets.load_dataset(revision=<commit>, streaming=True)`，不再使用会忽略 revision 的 datasets-server `/rows` 接口；本轮已有缓存以本地 SHA256 作为权威快照。

静态 packing 分析直接调用镜像中的 `greedy_knapsack`。当前 `DataArguments` 在 neat packing 时会把内部 knapsack 容量设为用户 cutoff 的 `cutoff-1`，processor 再补 1 个右侧 padding token；同时，静态模拟按正式配置的 8 个 `Dataset.map` worker 对连续分片分别装箱，与真实 processor 行为一致。静态结论只表示“允许进入成对实验”，不会直接开启 packing。最终开启条件仍是：

- no-packing 最佳 MBS=1：保守时间收益下界至少 10%；
- no-packing 最佳 MBS>1：保守时间收益下界至少 20%；
- 样本级期望 GBS 偏差不超过 5%。

## 4. 模型映射

本轮只比较同一 Qwen3 dense 架构下的 1.7B、4B、8B、14B，避免把开发机资源消耗在当前阶段不需要的端点上。

模型身份仍以配置中的原始权重目录名/路径为准。各 Qwen3 模型使用自身目录内 tokenizer；Qwen2.5-72B 显式使用 `/wanqing-models/Qwen3-8B` tokenizer。Qwen3 tokenizer 的最大 token id 为 151668，小于 72B 模型的 152064 embedding 容量。

| 名义量级 | 本地模型 | 实际参数量 | 最大上下文 | 当前 H800 阶段 | 后续硬件 |
|---:|---|---:|---:|---|---|
| 0.5B | Qwen3-0.6B | 0.752B | 40960 | 不运行 | RTX 4090 |
| 1.5B | Qwen3-1.7B | 2.032B | 40960 | FULL + LoRA | 补 RTX 4090 |
| 4B | Qwen3-4B | 4.023B | 40960 | FULL + LoRA | 按需要补测 |
| 7B | Qwen3-8B | 8.191B | 40960 | FULL + LoRA | 按需要补测 |
| 14B | Qwen3-14B | 14.768B | 40960 | FULL + LoRA | 按需要补测 |
| 32B | Qwen3-32B | 32.762B | 40960 | 延后 | 后续 H800 阶段 |
| 72B | Qwen2.5-72B | 72.706B | 131072 | 延后 | 8 卡或多机 |

## 5. 分阶段实验，而不是完整笛卡尔积

### 阶段 A：显存边界

- 186 个边界族：当前四模型产生 96 个基本网格角色，并为 8B/14B 增加 GPU、ZeRO 对照；去重后为 186。
- 每个边界族从 MBS=1 开始按 2 倍递增，第一个 OOM 后立即停止。
- 只有明确分类为 `oom` 的 trial 才能定义边界；资源忙会等待并重新排队，普通启动/代码失败不会写成显存边界标签。
- GBS 固定 64；GA 由 `64 / (DP × MBS)` 派生。
- 记录 allocated/reserved 峰值、nvidia-smi 峰值、OOM 阶段和错误类型。
- 1/2 卡边界探针可在不重叠 GPU 上并行；4 卡探针独占当前 1–4 卡池。

### 阶段 B：吞吐初筛、正式复验与 MFU

- 显存结果出来后才物化任务。每个固定的模型、训练类型、数据集和 GBS 场景，先保留每个可行 GPU 数的静态优选项，再补充 ZeRO/GC/MBS 对照，最多 4 个候选。
- 静态优选优先关闭 GC、优先 ZeRO-2 并使用较大兼容 MBS；仍保留策略对照，由短测决定实际吞吐排序，而不是直接把静态规则当成结论。
- 初筛对每个候选运行 warmup 2 + 稳定测量 4 optimizer steps。每个场景保留最低资源下最快的配置和全局最快配置 `Top-2`。
- 只有入围的 Top-2 运行 warmup 3 + 稳定测量 10 optimizer steps。最终推荐、epoch 时间和 GPU-hours 只使用正式测量；短测结果不会混入最终吞吐聚合。
- 每个入围物理配置默认正式测量 1 次；失败或健康检查异常才重跑。已有健康正式结果可替代同一物理配置的初筛。repeat、throughput 与 scaling 中重复出现的同一物理配置在推荐阶段合并统计，不重复计为候选。
- 正式吞吐在 GPU 1–4 内按不相交 mask 并行，以优先提高 GPU 利用率。不同任务会共享 CPU、存储、互联和功耗条件，因此报告需保留并行共享节点这一测量条件。
- 记录 micro/optimizer step 时间、计算/有效 tokens/s、samples/s、MFU、有效 MFU、功耗与时钟。

### 阶段 C：强扩展

- 12 个配置族：8B/14B × FULL/LoRA × 512/4096/32768。
- 固定模型、数据、cutoff、GBS=64，比较 1→2→4 卡。
- 每个卡数只测一个静态优选策略，运行 warmup 2 + 测量 8 optimizer steps，不重复扫描 ZeRO/GC/MBS。
- 每一级吞吐增益必须至少 70%；第一次不足 70% 后停止继续加卡。

### 阶段 D：packing 成对实验

- 20 个候选族，均已按 8-worker 真实装箱行为筛选，并满足对应 DP/GBS 下的期望样本 GBS 偏差 ≤5%。
- 先用 3 steps 验证 neat packing 的物理 MBS=1 显存可行性，再以 warmup 2 + 测量 8 steps 运行严格成对的 no-packing / packing 吞吐实验。
- 两侧保持模型、GPU、GC、ZeRO、目标 GBS、cutoff、数据顺序和 epoch 口径一致。

### 阶段 E：Profiler 校准与留出验证

- 8B 的 FULL/LoRA、GC on/off 共 4 个 profiler 小样本，使用 warmup 1 + 测量 3 steps 校准结构化 FLOP 公式。
- 留出未用于拟合的模型尺度/数据/GBS 组合，验证显存误差、时间误差、候选排序和 packing 不倒车。

## 6. MFU 口径

- 采集器按真实 batch 记录线性 token 数与注意力 `ΣL²`，而不是统一用 cutoff 或 `6×参数量×token`。
- FULL：线性、attention、lm_head 按 forward + dX + dW 计量。
- LoRA：冻结 base linear 按 forward + dX 计量，LoRA A/B 按完整训练计量，attention backward 单独计算；GC 重算不计入“有用 FLOPs”。
- 主 MFU 分母使用本机实测配置上限：132 SM × 4096 dense BF16 FLOPs/cycle/SM × 1.980GHz = 1.07085824 PFLOPS/卡。
- 另保留 989.5 TFLOPS/卡口径的 `public_reference_mfu` 便于跨报告比较；不会把公开页面标注为 sparsity 的 1,979 TFLOPS 当作 dense 峰值。
- 解析 FLOP 标签在 profiler 校准完成前标记为 `analytic_v1`。

## 7. GPU 卡池与共享节点调度

- 所有阶段允许：4×单卡、2×双卡、或 1×双卡 + 2×单卡并行；只有四卡任务独占 GPU 1–4。
- 当前双卡使用连续集合 `1,2` 或 `3,4`。
- 正式吞吐/强扩展/packing/profiler 优先使用不相交集合：单卡为 `1`/`2`/`3`/`4`，双卡为 `1,2`/`3,4`，四卡为 `1,2,3,4`。
- 所有任务只检查并使用 GPU 1–4；GPU 0、5、6、7 不在实验资源范围内，不会因其繁忙而阻塞，也不会被本实验抢占或终止。
- 当前阶段 `max_gpu_count=4`，调度器和单任务入口都会拒绝 8 卡任务或任何包含池外 GPU ID 的 mask。
- 共享机上若目标卡在 job 启动前临时被占用，scheduler 会保留原任务并延迟重试，不会把资源冲突记为 OOM 或不可行配置。
- 恢复运行可显式使用 `--join-busy-pool`：先在 GPU 1–4 中当前空闲的卡启动，并在忙卡释放后自动加入，不会终止或抢占已有进程。
- 审批失效或 launcher 未产生状态文件属于控制面致命错误：当前任务重新入队、停止发射新任务并等待已启动任务结束，绝不把启动失败当作已完成任务吞掉。

## 8. 开跑闸门

`scheduler.py` 默认只生成 dry-run。真正执行同时要求：

1. 人工确认本设计；
2. 批准文件绑定当前 `runtime/approval_design.json` 的 SHA256；
3. 当前配置卡池 GPU 1–4 空闲；不要求实验池外 GPU 空闲；
4. 真实 processor 对齐、模型目录/结构检查、Qwen3 tokenizer 兼容性、四组 smoke、运行时复现指纹和静态 preflight 全部通过；
5. 执行时冻结清单仍与当前文件逐项一致。

在收到明确批准前，不创建批准文件，也不启动任何训练。

## 9. 2026-07-22 决策附录：历史证据恢复与 109-row 计划退役

本附录记录实验完成后的决策，不修改前文的预注册设计和当时的执行口径。

对 1143 条 canonical H800 终态记录进行来源绑定的追溯恢复后，展示分类为
`legacy_verified=97`、`legacy_consistent=1014`、`diagnostic_only=31`、`rejected=1`。
恢复报告允许现有 H800 证据进入有边界的离线拟合，但不自动证明系数可发布；当前状态
为 `fit_decision=historical_component_scoped_fit_allowed`、
`ready_for_full_bounded_fit_bundle=true`、`calibration_publishable=false`。可拟合输入按用途
分别为 1062 条 feasibility、594 条 memory boundary、203 条正式 throughput 和 19 条
Profiler；241 条 throughput screen 不得冒充正式吞吐。10 个 packing pair 不是 ABBA，
只支持低置信效应估计，不能自动开启 packing。

因此，原先为“全部历史记录不可用”这一假设设计的 109-row 补跑方案不再是当前实验
计划。其 source candidate 和 frozen approval design schema
`sft_h800_calibration_candidate/v1`、
`sft_h800_calibration_approval_freeze/v1` 自本决策起永久退役，只作为审计材料保留，
不得重新生成、冻结、批准或执行，也不删除既有 artifact。

接下来的 H800 工作在无 GPU 环境中完成：以显存组成、优化器/梯度/激活、数据并行与
通信等理论项为主体，用恢复的历史测量校准少量系数，并按
`(model_id, train_type, dataset_id, target_gbs)` 场景分组做历史留出验证。推荐必须遵守
95% 显存容量上界和零 unsafe holdout OOM；吞吐只用于校准候选排序与卡数收益，不以
纯数值拟合代替机制约束。当前分区必须把同一场景在所有 runtime cohort 中的记录整体
留出，得到 84 个全局 resource folds（memory boundary 48、正式 throughput 78）；此前
按 cohort 分开的 214 个 folds 会让同一场景经由另一 cohort 泄漏进训练集，现仅保留为
诊断统计。旧 78-fold resource 指标也没有绑定当前 membership，只能作参考。旧 Profiler
辅助拟合的绑定 holdout MAPE 不等于新理论 planner 的验收结果。历史一致性证据必须扩大
不确定性，多个 runtime cohort 必须作为折内 nuisance/fixed effect，而不是拆分 holdout。

只有上述离线验证暴露无法由现有证据覆盖的具体缺口时，才允许提出新的最小前瞻实验。
新实验必须使用新 schema，绑定 historical recovery 与 readiness 报告的精确 SHA、
声明缺口、预算、提前停止和验收规则，并重新走人工审批。4090 数据继续独立处理，
不会在其流水线完成前混入 H800 拟合或发布验收。
